import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from methods.method import Method
from swa_gaussian.swag.posteriors import SWAG
import os

class Swag(Method):
    def __init__(self, config):
        super(Swag, self).__init__(config)
        self.init_model()
        self.swag_model = SWAG(
            self.model.__class__,
            no_cov_mat=False,
            max_num_models=20,
            config=config
        )

    def train_epoch(self, loader, criterion, cuda=True):
        loss_sum = 0.0
        correct = 0.0
        verb_stage = 0

        num_objects_current = 0
        num_batches = len(loader)

        self.model.train()

        for i, (input, target) in enumerate(loader):
            if cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            output = self.model(input)
            loss = criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.data.item() * input.size(0)

            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

            num_objects_current += input.size(0)

        return {
            "loss": loss_sum / num_objects_current,
            "accuracy": correct / num_objects_current * 100.0,
        }

    def eval(self, loader, model, criterion, cuda=True):
        loss_sum = 0.0
        correct = 0.0
        num_objects_total = len(loader.dataset)

        model.eval()

        with torch.no_grad():
            for i, (input, target) in enumerate(loader):
                if cuda:
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                output = model(input)
                loss = criterion(output, target)

                loss_sum += loss.item() * input.size(0)

                pred = output.data.argmax(1, keepdim=True)
                correct += pred.eq(target.data.view_as(pred)).sum().item()

        return {
            "loss": loss_sum / num_objects_total,
            "accuracy": correct / num_objects_total * 100.0,
        }

    def predict(self, loader):
        predictions = list()
        targets = list()

        self.model.eval()

        offset = 0
        with torch.no_grad():
            for input, target in loader:
                # input = input.cuda(non_blocking=True)
                output = self.model(input)

                batch_size = input.size(0)
                predictions.append(F.softmax(output, dim=1).cpu().numpy())
                targets.append(target.numpy())
                offset += batch_size

        return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}

    def _check_bn(self, module, flag):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            flag[0] = True

    def check_bn(self, model):
        flag = [False]
        model.apply(lambda module: self._check_bn(module, flag))
        return flag[0]

    def reset_bn(self, module):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)

    def _set_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]

    def _get_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum

    def bn_update(self, loader, model, **kwargs):
        """
            BatchNorm buffers update (if any).
            Performs 1 epochs to estimate buffers average using train dataset.

            :param loader: train dataset loader for buffers average estimation.
            :param model: model being update
            :return: None
        """
        if not self.check_bn(model):
            return
        model.train()
        momenta = {}
        model.apply(self.reset_bn)
        model.apply(lambda module: self._get_momenta(module, momenta))
        n = 0
        num_batches = len(loader)

        with torch.no_grad():
            for input, _ in loader:
                input_var = torch.autograd.Variable(input)
                b = input_var.data.size(0)

                momentum = b / (n + b)
                for module in momenta.keys():
                    module.momentum = momentum

                model(input_var, **kwargs)
                n += b

        model.apply(lambda module: self._set_momenta(module, momenta))

    def build_method(self, rebuild=False, **kwargs):
        if not rebuild:
            self.swag_model.load_state_dict(torch.load(os.path.join(self.config.method.output.path, 'model', 'swag_model.pt')))
            return
        self.train_method(kwargs['train_dl'], kwargs['valid_dl'])
        output_save_dir = os.path.join(self.config.method.output.path, 'model')
        os.makedirs(output_save_dir, exist_ok=True)
        torch.save(self.swag_model.state_dict(), os.path.join(output_save_dir, 'swag_model.pt'))


    def train_method(self, train_dl, valid_dl):
        self.train_dl = train_dl

        criterion = nn.CrossEntropyLoss()
        sgd_ens_preds = None
        sgd_targets = None
        n_ensembled = 0.0
        for epoch in range(self.config.method.epochs):

            lr = 0.01

            train_res = self.train_epoch(train_dl, criterion, cuda=False)

            test_res = self.eval(valid_dl, self.model, criterion, cuda=False)
            if (epoch + 1) > self.config.method.swag.swa_start:
                # sgd_preds, sgd_targets = utils.predictions(loaders["test"], model)
                sgd_res = self.predict(valid_dl)
                sgd_preds = sgd_res["predictions"]
                sgd_targets = sgd_res["targets"]
                print("updating sgd_ens")
                if sgd_ens_preds is None:
                    sgd_ens_preds = sgd_preds.copy()
                else:
                    # TODO: rewrite in a numerically stable way
                    sgd_ens_preds = sgd_ens_preds * n_ensembled / (
                            n_ensembled + 1
                    ) + sgd_preds / (n_ensembled + 1)
                n_ensembled += 1
                self.swag_model.collect_model(self.model)

                self.swag_model.sample(0.0)
                self.bn_update(train_dl, self.swag_model)
                swag_res = self.eval(valid_dl, self.swag_model, criterion, cuda=False)
                print(f"epoch: {epoch+1}/{self.config.method.epochs}, swag_loss: {swag_res['loss']}, swag_acc: {swag_res['accuracy']}")

            print(f"epoch: {epoch+1}/{self.config.method.epochs}, train_loss: {train_res["loss"]}, train_acc: {train_res["accuracy"]}, test_loss: {test_res["loss"]}, test_acc: {test_res['accuracy']}")

    def measure_uncertainty(self, loader, train_loader):

        import numpy as np
        import torch.nn.functional as F
        from tqdm import tqdm

        eps = 1e-12
        n_classes = 10
        n_samples = self.config.method.sample_size
        n_data = len(loader.dataset)

        # store per-sample predictive probabilities
        all_probs = np.zeros((n_samples, n_data, n_classes))

        for i in range(n_samples):
            self.bn_update(train_loader, self.swag_model)
            self.swag_model.eval()

            k = 0
            for input, target in tqdm(loader):
                self.swag_model.sample(scale=0.5, cov=True)
                torch.manual_seed(i)
                with torch.no_grad():
                    output = self.swag_model(input)
                    probs = F.softmax(output, dim=1).cpu().numpy()
                    all_probs[i, k:k + input.size(0), :] = probs
                k += input.size(0)

        # mean predictive probabilities
        mean_probs = np.mean(all_probs, axis=0)  # shape: (N, C)

        # --- Total uncertainty (predictive entropy)
        total_uncertainty = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)

        # --- Aleatoric uncertainty (expected entropy)
        aleatoric_uncertainty = -np.mean(
            np.sum(all_probs * np.log(all_probs + eps), axis=2),
            axis=0
        )

        # --- Epistemic uncertainty
        epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

        return {
            "total_uncertainty": total_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "out_of_distribution": 0,
        }
