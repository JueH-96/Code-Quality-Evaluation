from collections import Counter
from itertools import combinations
from typing import List

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        count = 0
        
        # Count the frequency of each number in nums
        freq = Counter(nums)
        
        # Generate all combinations of size 5
        for comb in combinations(nums, 5):
            # Count the frequency of each number in the combination
            comb_freq = Counter(comb)
            # Find the mode and its frequency
            mode, mode_count = max(comb_freq.items(), key=lambda x: x[1])
            # Check if the mode is unique
            if list(comb_freq.values()).count(mode_count) == 1:
                # Check if the middle element is the mode
                if comb[2] == mode:
                    count = (count + 1) % MOD
        
        return count