from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + nums[i - 1]
        
        prev_dp = [0] * (n + 1)
        
        for j in range(1, k + 1):
            # Determine the sign based on whether j is odd or even
            sign = 1 if (j % 2 == 1) else -1
            c_j = (k - j + 1) * sign
            
            current_max = prev_dp[0] - c_j * prefix[0]
            curr_dp = [0] * (n + 1)
            
            for i in range(1, n + 1):
                curr_dp[i] = c_j * prefix[i] + current_max
                # Calculate the candidate value for the next step
                candidate = prev_dp[i] - c_j * prefix[i]
                if candidate > current_max:
                    current_max = candidate
            
            prev_dp = curr_dp
        
        return max(prev_dp)