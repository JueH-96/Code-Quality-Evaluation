class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[j] = number of ways where arr1[current_position] = j
        dp = [0] * 51
        
        # Base case: first position
        for j in range(nums[0] + 1):
            dp[j] = 1
            
        # Process each subsequent position
        for i in range(1, n):
            new_dp = [0] * 51
            
            for j in range(nums[i] + 1):  # arr1[i] = j
                # For valid transitions from previous position
                for prev_j in range(nums[i-1] + 1):  # arr1[i-1] = prev_j
                    # Check constraints:
                    # 1. arr1[i] >= arr1[i-1]: j >= prev_j
                    # 2. arr2[i] <= arr2[i-1]: (nums[i] - j) <= (nums[i-1] - prev_j)
                    #    which means: prev_j <= nums[i-1] - nums[i] + j
                    
                    if j >= prev_j and prev_j <= nums[i-1] - nums[i] + j:
                        new_dp[j] = (new_dp[j] + dp[prev_j]) % MOD
            
            dp = new_dp
        
        return sum(dp) % MOD