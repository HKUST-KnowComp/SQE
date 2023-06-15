import torch
from torch.nn import Parameter
from torch.optim import Adam
import torch.optim
from torch import nn


#HELPER FUNCTIONS

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)
    
def artanh(x):
    return Artanh.apply(x)


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()

#EMBEDDING CLASS
class Manifold(object):
    """
    Abstract class to define operations on a manifold.
    """

    def __init__(self):
        super().__init__()
        self.eps = 10e-8

    def sqdist(self, p1, p2):
        """Squared distance between pairs of points."""
        raise NotImplementedError

    def egrad2rgrad(self, p, dp):
        """Converts Euclidean Gradient to Riemannian Gradients."""
        raise NotImplementedError

    def proj(self, p):
        """Projects point p on the manifold."""
        raise NotImplementedError

    def proj_tan(self, u, p):
        """Projects u on the tangent space of p."""
        raise NotImplementedError

    def proj_tan0(self, u):
        """Projects u on the tangent space of the origin."""
        raise NotImplementedError

    def expmap(self, u, p):
        """Exponential map of u at point p."""
        raise NotImplementedError

    def logmap(self, p1, p2):
        """Logarithmic map (base e) of point p1 at point p2."""
        raise NotImplementedError

    def expmap0(self, u):
        """Exponential map of u at the origin."""
        raise NotImplementedError

    def logmap0(self, p):
        """Logarithmic map of point p at the origin."""
        raise NotImplementedError

    def mobius_add(self, x, y, dim=-1):
        """Adds points x and y."""
        raise NotImplementedError

    def mobius_matvec(self, m, x):
        """Performs hyperboic martrix-vector multiplication."""
        raise NotImplementedError

    def init_weights(self, w, irange=1e-5):
        """Initializes random weigths on the manifold."""
        raise NotImplementedError

    def inner(self, p, u, v=None, keepdim=False):
        """Inner product for tangent vectors at point x."""
        raise NotImplementedError

    def ptransp(self, x, y, u):
        """Parallel transport of u from x to y. Refer to Eq. (3) in the paper."""
        raise NotImplementedError

    def ptransp0(self, x, u):
        """Parallel transport of u from the origin to y. Refer to Eq. (3) in the paper."""
        raise NotImplementedError


class ManifoldParameter(Parameter):
    """
    Subclass of torch.nn.Parameter for Riemannian optimization.
    """
    def __new__(cls, data, manifold, requires_grad = True):
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, manifold, requires_grad = True):
        self.manifold = manifold
        self.data = self.manifold.proj(data)

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()

    def __add__(self, b):
        return self.manifold.mobius_add(self.data,b)
    
    def __sub__(self, b):
        return self.manifold.mobius_add(self.data,-b)

class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.
    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c
    Note that 1/sqrt(c) is the Poincare ball radius.
    """

    def __init__(self, c = 1.):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.c = c
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2):
        sqrt_c = self.c ** 0.5
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def _lambda_x(self, x):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / (1. - self.c * x_sqnorm).clamp_min(self.min_norm)

    def egrad2rgrad(self, p, dp):
        lambda_p = self._lambda_x(p)
        dp /= lambda_p.pow(2)
        return dp

    def proj(self, x):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (self.c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def proj_tan(self, u, p):
        return u

    def proj_tan0(self, u):
        return u

    def expmap(self, u, p):
        sqrt_c = self.c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term)
        return gamma_1

    def logmap(self, p1, p2):
        sub = self.mobius_add(-p1, p2)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1)
        sqrt_c = self.c ** 0.5
        return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm

    def expmap0(self, u):
        sqrt_c = self.c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return gamma_1

    def logmap0(self, p):
        sqrt_c = self.c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x):
        sqrt_c = self.c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond, res_0, res_c)
        return res

    def init_weights(self, w, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = self.c ** 2
        a = -c2 * uw * v2 + self.c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - self.c * uw
        d = 1 + 2 * self.c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u):
        lambda_x = self._lambda_x(x)
        lambda_y = self._lambda_x(y)
        return self._gyration(y, -x, u) * lambda_x / lambda_y

    def ptransp_(self, x, y, u):
        lambda_x = self._lambda_x(x)
        lambda_y = self._lambda_x(y)
        return self._gyration(y, -x, u) * lambda_x / lambda_y

    def ptransp0(self, x, u):
        lambda_x = self._lambda_x(x)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

    def to_lorentz(self, x):
        K = 1./ self.c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
        return sqrtK * torch.cat([K + sqnorm, 2 * sqrtK * x], dim=1) / (K - sqnorm)

#OPTIMIZER: RADAM
default_manifold = PoincareBall()

class OptimMixin(object):
    def __init__(self, *args, stabilize=None, **kwargs):
        self._stabilize = stabilize
        super().__init__(*args, **kwargs)

    def stabilize_group(self, group):
        pass

    def stabilize(self):
        """Stabilize parameters if they are off-manifold due to numerical reasons
        """
        for group in self.param_groups:
            self.stabilize_group(group)


def copy_or_set_(dest, source):
    """
    A workaround to respect strides of :code:`dest` when copying :code:`source`
    (https://github.com/geoopt/geoopt/issues/70)
    Parameters
    ----------
    dest : torch.Tensor
        Destination tensor where to store new data
    source : torch.Tensor
        Source data to put in the new tensor
    Returns
    -------
    dest
        torch.Tensor, modified inplace
    """
    if dest.stride() != source.stride():
        return dest.copy_(source)
    else:
        return dest.set_(source)


class RiemannianAdam(OptimMixin, torch.optim.Adam):
    r"""Riemannian Adam with the same API as :class:`torch.optim.Adam`
    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float (optional)
        learning rate (default: 1e-3)
    betas : Tuple[float, float] (optional)
        coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps : float (optional)
        term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    amsgrad : bool (optional)
        whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False)
    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def step(self, manifold = PoincareBall(), closure=None):
        """Performs a single optimization step.
        Arguments
        ---------
        closure : callable (optional)
            A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                if "step" not in group:
                    group["step"] = 0
                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]

                amsgrad = group["amsgrad"]
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue
                    if isinstance(point, (ManifoldParameter)):
                         manifold = point.manifold
                         c = point.manifold.c
                    manifold = manifold
                    c = manifold.c
                    if grad.is_sparse:
                        raise RuntimeError(
                                "Riemannian Adam does not support sparse gradients yet. Did not need in my experiments."
                        )

                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(point)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(point)
                    # make local variables for easy access
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    # actual step
                    grad.add_(weight_decay, point)
                    grad = manifold.egrad2rgrad(point, grad)
                    exp_avg.mul_(betas[0]).add_(1 - betas[0], grad)
                    exp_avg_sq.mul_(betas[1]).add_(
                            1 - betas[1], manifold.inner(point, grad, keepdim=True)
                    )
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.sqrt().add_(eps)
                    else:
                        denom = exp_avg_sq.sqrt().add_(eps)
                    group["step"] += 1
                    bias_correction1 = 1 - betas[0] ** group["step"]
                    bias_correction2 = 1 - betas[1] ** group["step"]
                    step_size = (
                        learning_rate * bias_correction2 ** 0.5 / bias_correction1
                    )

                    # copy the state, we need it for retraction
                    # get the direction for ascend
                    direction = exp_avg / denom
                    # transport the exponential averaging to the new point
                    new_point = manifold.proj(manifold.expmap(-step_size * direction, point))
                    exp_avg_new = manifold.ptransp(point, new_point, exp_avg)
                    # use copy only for user facing point
                    copy_or_set_(point, new_point)
                    exp_avg.set_(exp_avg_new)

                    group["step"] += 1
                if self._stabilize is not None and group["step"] % self._stabilize == 0:
                    self.stabilize_group(group)
        return loss

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, ManifoldParameter):
                continue
            state = self.state[p]
            if not state:  # due to None grads
                continue
            manifold = p.manifold
            c = p.c
            exp_avg = state["exp_avg"]
            copy_or_set_(p, manifold.proj(p))
            exp_avg.set_(manifold.proj_tan(exp_avg))
