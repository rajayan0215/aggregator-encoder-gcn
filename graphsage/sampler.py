import random


class RandomSampler:

    def __init__(self, num_samples=10):
        """

        Args:
            num_samples (int, optional):
        """

        super(RandomSampler, self).__init__()
        self.num_samples = num_samples

    def sample(self, neighbourhoods, black_list=[]):
        """
        Sampling without replacement

        Args:
            neighbourhoods () :
            num_sample (int) : Number of items to sample from the 2nd dimension
            black_list (set, optional) : items to exclude
        """
        # local function pointers as a speed hack
        _set = set
        _sample = random.sample
        has_black_list = len(black_list) <= 0

        result = []
        for i in range(len(neighbourhoods)):
            neighbourhood = _set(neighbourhoods[i]) if has_black_list else _set(neighbourhoods[i]) - _set(black_list[i])
            result.append(
                _sample(neighbourhood, self.num_samples) if self.num_samples < len(neighbourhood) else neighbourhood)

        return result


class HybridSampler:

    def __init__(self, priority_list, num_samples=10):
        """

        Args:
            num_samples (int, optional):
        """

        super(HybridSampler, self).__init__()
        self.num_samples = num_samples
        dim = int(self.num_samples / 2)
        self.rs = RandomSampler(dim)
        self.ps = PrioritySampler(priority_list, dim)

    def sample(self, neighbourhoods):
        """
        Sampling without replacement

        Args:
            neighbourhoods () :
            priority_list () :
            num_sample (int) : Number of items to sample from the 2nd dimension
        """

        important_ones = self.ps.sample(neighbourhoods)
        random_ones = self.rs.sample(neighbourhoods, important_ones)
        return [list(important_ones[i]) + list(random_ones[i]) for i in range(len(neighbourhoods))]


class PrioritySampler:

    def __init__(self, priority_list, num_samples=10):
        """

        Args:
            num_samples (int, optional):
        """

        super(PrioritySampler, self).__init__()
        self.num_samples = num_samples
        self.priority_list = priority_list

    def sample(self, neighbourhoods, black_list=[]):
        """
        Sampling without replacement

        Args:
            neighbourhoods () :
            num_sample (int) : Number of items to sample from the 2nd dimension
            black_list (set, optional) : items to exclude
        """
        # local function pointers as a speed hack
        _set = set
        no_exceptions = len(black_list) <= 0

        result = []
        for i in range(len(neighbourhoods)):
            neighbourhood = _set(neighbourhoods[i]) if no_exceptions else _set(neighbourhoods[i]) - _set(black_list[i])
            result.append(self.pick(neighbourhood) if self.num_samples < len(neighbourhood) else neighbourhood)

        return result

    def pick(self, neighbourhood):
        sample = []
        for i in range(len(self.priority_list)):
            if self.priority_list[i] in neighbourhood:
                sample.append(self.priority_list[i])

        return sample
