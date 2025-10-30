from typing import List

class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        dp = []
        max_after_d = []
        max_length = 0  # The maximum possible length is at least 1, but with constraints, it's 2
        
        for j in range(n):
            current_dp = [0] * 300  # Since maximum possible difference is 299
            for i in range(j):
                diff = abs(nums[j] - nums[i])
                if diff >= 300:
                    continue  # Impossible since nums[i] and nums[j] are <=300
                # Get the maximum length from previous subsequences ending at i with difference >= diff
                max_prev = max_after_d[i][diff]
                candidate = max_prev + 1
                # Ensure at least a pair (length 2)
                if candidate < 2:
                    candidate = 2
                if candidate > current_dp[diff]:
                    current_dp[diff] = candidate
            
            # Compute the suffix maximum array for current_dp
            current_max_after = [0] * 300
            max_so_far = 0
            for d in range(299, -1, -1):
                max_so_far = max(max_so_far, current_dp[d])
                current_max_after[d] = max_so_far
            
            dp.append(current_dp)
            max_after_d.append(current_max_after)
            
            # Update the overall maximum length
            current_max = max(current_dp)
            if current_max > max_length:
                max_length = current_max
        
        return max_length