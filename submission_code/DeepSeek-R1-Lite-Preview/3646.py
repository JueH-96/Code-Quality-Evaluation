from typing import List

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        if not nums:
            return 0
        max_num = max(nums)
        # Initialize dp and count arrays with 0
        dp = [0] * (max_num + 2)
        count = [0] * (max_num + 2)
        total_sum = 0
        for num in nums:
            # Ensure num-1 and num+1 are within bounds
            left = num - 1
            right = num + 1
            count_left = count[left] if left >= 0 else 0
            dp_left = dp[left] if left >= 0 else 0
            count_right = count[right] if right <= max_num + 1 else 0
            dp_right = dp[right] if right <= max_num + 1 else 0
            # Update count and dp for current num
            current_count = 1 + count_left + count_right
            current_dp = num + dp_left + dp_right + count_left * num + count_right * num
            count[num] = current_count
            dp[num] = current_dp
            # Add to total_sum
            total_sum = (total_sum + current_dp) % MOD
        return total_sum