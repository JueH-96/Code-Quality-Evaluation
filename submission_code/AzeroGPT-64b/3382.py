from itertools import accumulate

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        def compute_subarrays(nums, x):
            count = 0
            dp = [0] * (len(nums) + 1)
            for i, num in enumerate(nums):
                if num == x:
                    dp[i + 1] = i + 1
                else:
                    dp[i + 1] = dp[i]
                count += dp[i]
            return count
        
        max_val = max(nums)
        return compute_subarrays(nums, max_val) + compute_subarrays(nums[::-1], max_val) - len(nums)