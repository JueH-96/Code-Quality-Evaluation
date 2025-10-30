from typing import List

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = {}
        for num in nums:
            val1 = num
            val2 = num + 1
            # Calculate new possible lengths for val1 and val2
            prev1 = dp.get(val1 - 1, 0)
            new_len1 = prev1 + 1
            prev2 = dp.get(val2 - 1, 0)
            new_len2 = prev2 + 1
            # Update the dp dictionary
            if new_len1 > dp.get(val1, 0):
                dp[val1] = new_len1
            if new_len2 > dp.get(val2, 0):
                dp[val2] = new_len2
        return max(dp.values()) if dp else 0