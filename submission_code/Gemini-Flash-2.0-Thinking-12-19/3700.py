import itertools
from collections import Counter
from typing import List

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        mod = 10**9 + 7
        for mid_index in range(2, n - 2):
            prefix_indices_combinations = itertools.combinations(range(mid_index), 2)
            suffix_indices_combinations = itertools.combinations(range(mid_index + 1, n), 2)
            for prefix_indices in prefix_indices_combinations:
                for suffix_indices in suffix_indices_combinations:
                    indices = list(prefix_indices) + [mid_index] + list(suffix_indices)
                    subsequence = [nums[i] for i in indices]
                    if self.has_unique_middle_mode(subsequence):
                        count = (count + 1) % mod
        return count

    def has_unique_middle_mode(self, seq):
        counts = Counter(seq)
        max_freq = 0
        for x in counts:
            max_freq = max(max_freq, counts[x])
        modes = []
        for x in counts:
            if counts[x] == max_freq:
                modes.append(x)
        if len(modes) == 1:
            return modes[0] == seq[2]
        return False