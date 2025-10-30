from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        dp = 1  # Bitmask where bit i is set if sum i is achievable
        rewardValues.sort()
        for num in rewardValues:
            if num == 0:
                continue
            eligible_mask = dp & ((1 << num) - 1)
            new_bits = eligible_mask << num
            dp |= new_bits
        return dp.bit_length() - 1