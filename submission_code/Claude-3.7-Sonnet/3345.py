class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        total_power = 0
        
        # Iterate through all possible subsequences using bitmasks
        for mask in range(1, 1 << n):
            # Create the current subsequence
            subset = []
            for i in range(n):
                if mask & (1 << i):
                    subset.append(nums[i])
            
            # Compute the power of this subsequence using knapsack DP
            dp = [0] * (k + 1)
            dp[0] = 1  # Empty subsequence has sum 0
            
            for num in subset:
                # We iterate backwards to avoid counting the same element multiple times
                for j in range(k, num - 1, -1):
                    dp[j] = (dp[j] + dp[j - num]) % MOD
            
            # Add the power of this subsequence to our total
            total_power = (total_power + dp[k]) % MOD
        
        return total_power