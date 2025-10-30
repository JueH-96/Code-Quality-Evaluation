from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute the coefficients for each of the k subarrays
        coefficients = [(k - j) * (-1)**j for j in range(k)]
        
        # Initialize prev_best to store the maximum strength up to the previous subarray count
        prev_best = [0] * n
        
        for j in range(k):
            current_dp = [-float('inf')] * n
            coeff = coefficients[j]
            for i in range(n):
                # Option 2: Start a new subarray at i
                if i == 0:
                    option2 = coeff * nums[i]
                else:
                    option2 = prev_best[i-1] + coeff * nums[i]
                
                # Option 1: Extend the previous subarray (if possible)
                if i > 0:
                    option1 = current_dp[i-1] + coeff * nums[i]
                    current_dp[i] = max(option1, option2)
                else:
                    current_dp[i] = option2
            
            # Update prev_best with the current_best (prefix max of current_dp)
            current_best = [ -float('inf') ] * n
            current_best[0] = current_dp[0]
            for i in range(1, n):
                current_best[i] = max(current_best[i-1], current_dp[i])
            prev_best = current_best
        
        return prev_best[-1]