from typing import List

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        prev_one = -1
        ways = 1
        count_ones = 0
        
        for i, v in enumerate(nums):
            if v == 1:
                count_ones += 1
                if prev_one != -1:
                    # distance between this 1 and the previous 1 gives (gap_zeros + 1) choices
                    ways = (ways * (i - prev_one)) % MOD
                prev_one = i
        
        # If there are no 1's, no valid split. Otherwise ways is correct (1 if exactly one 1).
        return ways if count_ones > 0 else 0