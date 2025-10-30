from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Precompute coefficients for each j (1-based)
        coeff = [0] * (k + 1)
        for j in range(1, k + 1):
            sign = 1 if j % 2 == 1 else -1
            coeff[j] = sign * (k - j + 1)
        
        # Initialize global_val and dp arrays
        global_val = [float('-inf')] * (k + 1)
        global_val[0] = 0
        dp = [float('-inf')] * (k + 1)
        
        for num in nums:
            new_dp = [float('-inf')] * (k + 1)
            new_global = list(global_val)  # Start with previous global values
            for j in range(1, k + 1):
                # Calculate option1: start new subarray
                if global_val[j-1] != float('-inf'):
                    option1 = global_val[j-1] + num * coeff[j]
                else:
                    option1 = float('-inf')
                # Calculate option2: extend previous subarray
                if dp[j] != float('-inf'):
                    option2 = dp[j] + num * coeff[j]
                else:
                    option2 = float('-inf')
                new_dp_j = max(option1, option2)
                new_dp[j] = new_dp_j
                # Update new_global[j]
                new_global[j] = max(new_dp_j, global_val[j])
            # Update dp and global_val for next iteration
            dp = new_dp
            global_val = new_global
        
        return global_val[k]