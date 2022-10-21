from ..builder.registry import register

from torch.utils.data import RandomSampler

# RandomSampler
# data_source: Sized, replacement: bool = False, num_samples: Optional[int] = None

@register(name = "SAMPLER")
class MultiRandomSampler(RandomSampler):
    def __init__(self, data_source, replacement = False, multiple_times = 1):
        super().__init__(data_source, replacement, int(round(multiple_times * len(data_source))))