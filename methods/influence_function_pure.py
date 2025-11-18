import torch
from methods.method import Method
from tqdm import tqdm

class InfluenceFunction(Method):
    def __init__(self, config):
        super().__init__(config)

    def _hvp(self, loss, v):
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])

        grad_v = torch.dot(flat_grads, v)
        hv = torch.autograd.grad(grad_v, self.model.parameters(), retain_graph=True)
        hv_flat = torch.cat([h.contiguous().view(-1) for h in hv])
        return hv_flat

    def _conjugate_gradient(self, loss, b, cg_iters=100, tol=1e-10, damp=0.01):
        torch.autograd.set_detect_anomaly(True)

        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)

        for i in range(cg_iters):

            # Hessian-vector product
            Av = self._hvp(loss, p) + damp * p

            alpha = rdotr / torch.dot(p, Av)

            # OUT-OF-PLACE updates
            x = x + alpha * p
            r = r - alpha * Av

            new_rdotr = torch.dot(r, r)
            if new_rdotr < tol:
                break

            beta = new_rdotr / rdotr

            p = r + beta * p
            rdotr = new_rdotr

        return x

    def __flatten_grads(self, grads):
        return torch.cat([g.contiguous().view(-1) for g in grads])

    def train_base_model(self, loader: torch.utils.data.DataLoader):
        pass

    def measure_uncertainty(self, loader: torch.utils.data.DataLoader):
        self.model.eval()
        output = []
        for x_test, y_test in tqdm(loader):
            self.model.zero_grad()
            out = self.model(x_test.to(self.device))
            loss = torch.nn.functional.cross_entropy(out, y_test.to(self.device))

            # g = ∇θ loss(x)
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            g = self.__flatten_grads(grads)

            # u = H^{-1} g   (IHVP)
            u = self._conjugate_gradient(loss, g)

            # uncertainty = gᵀ u
            uncertainty = torch.dot(g, u).item()
            output.append(uncertainty)

        output = torch.cat(output, dim=0)
        print(output.shape)

        return {
            "total_uncertainty": output,
            "epistemic_uncertainty": output,
            "all_uncertainty": 0,
            "out_of_distribution": 0
        }
