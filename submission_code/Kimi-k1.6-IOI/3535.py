class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize previous DP and prefix sum arrays
        prev_dp = [0] * 51  # Values can be up to 50
        first_num = nums[0]
        for v in range(first_num + 1):
            prev_dp[v] = 1
        
        # Compute prefix sums for the initial state
        prefix_sum_prev = [0] * 51
        prefix_sum_prev[0] = prev_dp[0] % MOD
        for i in range(1, 51):
            prefix_sum_prev[i] = (prefix_sum_prev[i-1] + prev_dp[i]) % MOD
        
        for i in range(1, n):
            curr_dp = [0] * 51
            current_max = nums[i]
            prev_max = nums[i-1]
            delta = max(0, nums[i] - nums[i-1])
            
            for v in range(current_max + 1):
                upper = v - delta
                upper = min(upper, prev_max)
                if upper < 0:
                    sum_u = 0
                else:
                    sum_u = prefix_sum_prev[upper] if upper <= 50 else 0
                curr_dp[v] = sum_u % MOD
            
            # Compute prefix sums for the current DP array
            prefix_sum_curr = [0] * 51
            prefix_sum_curr[0] = curr_dp[0] % MOD
            for j in range(1, 51):
                prefix_sum_curr[j] = (prefix_sum_curr[j-1] + curr_dp[j]) % MOD
            
            # Update previous DP and prefix sums for the next iteration
            prev_dp = curr_dp
            prefix_sum_prev = prefix_sum_curr
        
        # Sum all valid values in the last DP array up to nums[-1]
        total = sum(prev_dp[:nums[-1] + 1]) % MOD
        return total