from typing import List
from math import gcd
from functools import reduce
from collections import defaultdict

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)

        # Function to calculate GCD of a list
        def gcd_list(lst):
            return reduce(gcd, lst)

        # Dictionary to store the count of each GCD
        gcd_count = defaultdict(int)

        # Generate all non-empty subsequences using bitmask
        for mask in range(1, 1 << n):
            subsequence = [nums[i] for i in range(n) if (mask & (1 << i))]
            gcd_value = gcd_list(subsequence)
            gcd_count[gcd_value] += 1

        # Calculate the number of valid pairs
        result = 0
        for count in gcd_count.values():
            if count > 1:
                result += (count * (count - 1)) // 2
                result %= MOD

        return result