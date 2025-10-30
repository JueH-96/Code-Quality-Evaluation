class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize for the first element
        prev_max = nums[0]
        prev_dp = [0] * (prev_max + 1)
        for x in range(prev_max + 1):
            prev_dp[x] = 1
        
        # Compute the prefix sum array for the first step
        prefix_prev = [0] * (prev_max + 2)
        for x in range(prev_max + 1):
            prefix_prev[x + 1] = (prefix_prev[x] + prev_dp[x]) % MOD
        
        for i in range(1, n):
            m = nums[i]
            d = max(0, m - nums[i - 1])
            current_dp = [0] * (m + 1)
            
            # Compute the current dp values using the prefix sum from the previous step
            for x in range(m + 1):
                if x < d:
                    current_dp[x] = 0
                else:
                    required = x - d
                    if required > nums[i - 1]:
                        current_dp[x] = 0
                    else:
                        # Ensure we don't go out of bounds
                        if required + 1 > len(prefix_prev) - 1:
                            current_dp[x] = 0
                        else:
                            current_dp[x] = prefix_prev[required + 1] % MOD
            
            # Update the prefix sum for the next iteration
            prefix_current = [0] * (m + 2)
            for x in range(m + 1):
                prefix_current[x + 1] = (prefix_current[x] + current_dp[x]) % MOD
            
            # Update previous dp and prefix for the next iteration
            prev_dp = current_dp
            prefix_prev = prefix_current
        
        # Sum all possibilities for the last element
        total = sum(prev_dp) % MOD
        return total