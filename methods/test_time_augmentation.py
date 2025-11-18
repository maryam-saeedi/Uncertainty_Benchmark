from methods.method import Method
import torch
import torch.nn.functional as F
from torchvision.transforms import *
from tqdm import tqdm
import importlib

def build_transform(cfg):
    """Recursively build transforms (supports Compose, RandomChoice, etc.)."""
    class_path = cfg['_type']
    cls = globals()[class_path]

    # Handle nested transforms
    if 'transforms' in cfg:
        sub_transforms = [build_transform(t) for t in cfg['transforms']]
        kwargs = {k: v for k, v in cfg.items() if k not in ['_type', 'transforms']}
        return cls(sub_transforms, **kwargs)

    # Simple transform
    kwargs = {k: v for k, v in cfg.items() if k != '_type'}
    return cls(**kwargs)

class TTA(Method):
    def __init__(self, config):
        super(TTA, self).__init__(config)
        self.default_augmentation = build_transform(config.method.augmentation)
        print(self.default_augmentation)

    def measure_uncertainty(self, loader: torch.utils.data.DataLoader):
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for inputs_, targets_ in tqdm(loader):
                x = torch.cat([self.default_augmentation(inputs_) for _ in range(self.config.get('sample_size', 100))], dim=0)
                x = x.to(self.device)
                output = self.model(x)
                output = F.softmax(output, dim=-1)
                outputs.append(output.unsqueeze(0))

        predictions = torch.cat(outputs, dim=0)
        aleatoric_uncertainty = -torch.mean(torch.sum(predictions * torch.log(predictions), dim=2), dim=1)

        p_mean = torch.mean(predictions, dim=1)
        total_uncertainty = -torch.sum(p_mean * torch.log(p_mean), dim=1)

        epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

        return {
            "total_uncertainty": total_uncertainty.detach().cpu().numpy(),
            "aleatoric_uncertainty": aleatoric_uncertainty.detach().cpu().numpy(),
            "epistemic_uncertainty": epistemic_uncertainty.detach().cpu().numpy(),
            "out_of_distribution": 0,
        }