class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        max_val = max(nums)
        
        # dp[val] = number of ways to have arr1[current_index] = val
        dp = [0] * (max_val + 1)
        
        # Base case: for index 0
        for val in range(nums[0] + 1):
            dp[val] = 1
        
        for i in range(1, n):
            new_dp = [0] * (max_val + 1)
            
            # Compute prefix sums for current dp
            prefix = [0] * (max_val + 2)
            for j in range(max_val + 1):
                prefix[j + 1] = (prefix[j] + dp[j]) % MOD
            
            for val in range(nums[i] + 1):
                # Constraint: arr1[i-1] <= val - max(0, nums[i] - nums[i-1])
                # Also: arr1[i-1] <= nums[i-1]
                max_prev = min(nums[i-1], val - max(0, nums[i] - nums[i-1]))
                if max_prev >= 0:
                    new_dp[val] = prefix[max_prev + 1]
            
            dp = new_dp
        
        # Sum all valid values for the last index
        result = 0
        for val in range(nums[n-1] + 1):
            result = (result + dp[val]) % MOD
        
        return result