from typing import List
from collections import Counter

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        count = 0

        # Generate all subsequences of size 5
        from itertools import combinations
        for subsequence in combinations(nums, 5):
            counter = Counter(subsequence)
            middle = subsequence[2]
            max_freq = max(counter.values())

            # Check if the middle element is the unique mode
            if counter[middle] == max_freq and list(counter.values()).count(max_freq) == 1:
                count += 1

        return count % MOD