# https://www.johndcook.com/blog/standard_deviation/

class RunningStat:

    def __init__(self):
        self.m_n = 0

    def clear(self):
        self.m_n = 0

    def push(self, x):
        self.m_n += 1
        x = x.detach().clone()
        x.requires_grad = False
        if self.m_n == 1:
            self.m_oldM = self.m_newM = x + 0
            self.m_oldS = x * 0
            self.m_newS = x * 0
        else:
            self.m_newM = self.m_oldM + (x - self.m_oldM) / self.m_n
            self.m_newS = self.m_oldS + (x - self.m_oldM) * (x - self.m_newM)
            self.m_oldM = self.m_newM
            self.m_oldS = self.m_newS

    def num_data_values(self):
        return self.m_n

    def mean(self):
        return self.m_newM if self.m_n > 0 else 0.0

    def variance(self):
        return self.m_newS / (self.m_n - 1) if self.m_n > 1 else 0.0

    def standard_deviation(self):
        return self.variance() ** 0.5
    
    def reshape(self, newshape):
      self.m_oldS = self.m_oldS[newshape]
      self.m_oldM = self.m_oldM[newshape]
      self.m_newS = self.m_newS[newshape]
      self.m_newM = self.m_newM[newshape]

from functools import wraps
import numpy as np

class linear_pruner:

    def __init__(self, layerprev, layernext, activation=lambda x: x, model_and_optimizer = None):
        self.layerprev = layerprev
        self.layernext = layernext
        self.old_layerprev_forward = layerprev.forward
        self.activation = activation
        self.model_and_optimizer = model_and_optimizer # for momentum
        self.stats = RunningStat()

    def hook(self):
        @wraps(self.old_layerprev_forward)
        def new_layerprev_forward_observe(*_, **__):
            x = self.old_layerprev_forward(*_, **__)
            if len(x.shape) == 1:
                self.stats.push(x)
            elif len(x.shape) == 2:
                for i in x:
                    self.stats.push(i)
            else:
                raise NotImplementedError(f"Cannot override forward pass for input shape {x.shape}")
            return x
        self.layerprev.forward = new_layerprev_forward_observe
  
    def unhook(self):
        self.layerprev.forward = self.old_layerprev_forward

    def test_prune(self, threshold_mean_fraction: float = 0.6, threshold_percentile: int = 0):
        # Threshold of what needs pruning
        assert (threshold_mean_fraction >= 0) and (100 >threshold_percentile >= 0), "Need valid_vaues"
        if threshold_mean_fraction > 0:
            threshold = torch.mean(self.stats.standard_deviation())*threshold_mean_fraction
        else:
            threshold = np.percentile(np.array(self.stats.standard_deviation().detach().clone().cpu()), threshold_percentile)
        bad_locations = torch.where(self.stats.standard_deviation() < threshold)[0]
        meanreplacement = self.activation(self.stats.mean()[bad_locations])
        @wraps(self.old_layerprev_forward)
        def new_layerprev_forward_test(*_, **__):
            x = self.old_layerprev_forward(*_, **__)
            if len(x.shape) == 1:
               x[bad_locations] = meanreplacement
            elif len(x.shape) == 2:
               x[:, bad_locations] = meanreplacement
            else:
                raise NotImplementedError(f"Cannot override forward pass for input shape {x.shape}")
            return x
        self.layerprev.forward = new_layerprev_forward_test

    def permanent_prune(self, threshold_mean_fraction: float = 0.6, threshold_percentile: int = 0):
        # Threshold of what needs pruning
        assert (threshold_mean_fraction >= 0) and (100 >threshold_percentile >= 0), "Need valid_vaues"
        if threshold_mean_fraction > 0:
            threshold = torch.mean(self.stats.standard_deviation())*threshold_mean_fraction
        else:
            threshold = np.percentile(np.array(self.stats.standard_deviation().detach().clone().cpu()), threshold_percentile)
        # Get pruning locations
        bad_locations = torch.where(self.stats.standard_deviation() < threshold)[0]
        good_locations = torch.where(self.stats.standard_deviation() >= threshold)[0]

        if self.model_and_optimizer:
            state_dict = self.model_and_optimizer[1].state_dict()['state']
        # Prune layerprev and state
        if self.model_and_optimizer:
            index = next(i for i, p in enumerate(self.model_and_optimizer[0].parameters()) if p is self.layerprev.weight)
            state_dict[index]["momentum_buffer"].data = state_dict[index]["momentum_buffer"].data[good_locations, :]
        self.layerprev.weight.data = self.layerprev.weight.data[good_locations, :]
        if self.model_and_optimizer:
            index = next(i for i, p in enumerate(self.model_and_optimizer[0].parameters()) if p is self.layerprev.bias)
            state_dict[index]["momentum_buffer"].data = state_dict[index]["momentum_buffer"].data[good_locations]

        self.layerprev.bias.data = self.layerprev.bias.data[good_locations]
        self.layerprev.out_features = len(good_locations)

        # Prune layerprev grad
        if self.layerprev.weight.grad is not None:
            self.layerprev.weight.grad.data = self.layerprev.weight.grad.data[good_locations, :]
        if self.layerprev.bias.grad is not None:
            self.layerprev.bias.grad.data = self.layerprev.bias.grad.data[good_locations]

        # Edit layernext bias data
        temp_bias = self.layernext.bias.data + 0
        self.layernext.bias.data = temp_bias * 0
        dummy = torch.zeros(self.layernext.weight.data.shape[1], device=self.layernext.weight.data.device)
        dummy[bad_locations] = self.stats.mean()[bad_locations]
        dummy = self.activation(dummy)
        bias_edit = self.layernext.forward(dummy)
        self.layernext.bias.data = temp_bias + bias_edit

        # Prune layernext weight
        if self.model_and_optimizer:
            index = next(i for i, p in enumerate(self.model_and_optimizer[0].parameters()) if p is self.layernext.weight)
            state_dict[index]["momentum_buffer"].data = state_dict[index]["momentum_buffer"].data[:, good_locations]
        self.layernext.weight.data = self.layernext.weight.data[:, good_locations]
        self.layernext.in_features = len(good_locations)
        # Prune layernext grad
        if self.layernext.weight.grad is not None:
            self.layernext.weight.grad.data = self.layernext.weight.grad.data[:, good_locations]
        # reshape stats
        self.stats.reshape(good_locations)
        # Remake parameters
        self.layerprev.weight = nn.Parameter(self.layerprev.weight)
        self.layerprev.bias = nn.Parameter(self.layerprev.bias)
        self.layernext.weight = nn.Parameter(self.layernext.weight)

        #Force update optimizer params
        if self.model_and_optimizer:
            new_state = dict(zip(self.model_and_optimizer[0].parameters(), self.model_and_optimizer[1].state.values()))
            self.model_and_optimizer[1].state.clear()
            self.model_and_optimizer[1].state.update(new_state)
            self.model_and_optimizer[1].param_groups[0]['params'] = list(self.model_and_optimizer[0].parameters())
