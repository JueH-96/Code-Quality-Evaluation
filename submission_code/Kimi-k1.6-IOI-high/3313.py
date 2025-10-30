class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i-1] + nums[i-1]
        
        prev_dp = [0] * (n + 1)  # dp[0][i] = 0 for all i
        
        for j in range(1, k + 1):
            coeff = (-1) ** (j + 1) * (k - j + 1)
            curr_dp = [-float('inf')] * (n + 1)
            curr_dp[0] = -float('inf')
            max_prev = -float('inf')
            
            for i in range(1, n + 1):
                # Update max_prev with prev_dp[i-1] (l = i-1)
                candidate = prev_dp[i-1] - coeff * prefix_sum[i-1]
                if candidate > max_prev:
                    max_prev = candidate
                
                # Calculate the current options
                option1 = curr_dp[i-1] if (i - 1) >= 0 else -float('inf')
                option2 = coeff * prefix_sum[i] + max_prev if max_prev != -float('inf') else -float('inf')
                
                curr_dp[i] = max(option1, option2)
            
            prev_dp = curr_dp
        
        return prev_dp[n]