from typing import List
from collections import defaultdict
from math import sqrt

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        def is_perfect_square(n):
            return sqrt(n).is_integer()

        def get_subsets(s):
            return [s & (s - 1) for _ in range(s.bit_length())]

        def get_sum(s):
            return sum(nums[i] for i in range(n) if (s >> i) & 1)

        n = len(nums)
        subsets = defaultdict(list)
        for i in range(1, 1 << n):
            subsets[bin(i).count('1')].append(i)

        max_sum = -1
        for count in sorted(subsets, reverse=True):
            for s in subsets[count]:
                if all(is_perfect_square(get_sum(s & t)) for t in get_subsets(s)):
                    max_sum = max(max_sum, get_sum(s))
                    break
            if max_sum != -1:
                break

        return max_sum