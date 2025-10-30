class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        max_val = max(nums)
        
        # dp[i][v] will store the number of ways to form a non-decreasing sequence 
        # up to index i with i-th value being exactly v
        dp = [[0] * (max_val + 1) for _ in range(n)]
        
        # Initialize dp for the first element
        for v in range(nums[0] + 1):
            dp[0][v] = 1
        
        # Fill dp for subsequent elements
        for i in range(1, n):
            # Accumulate sum to maintain the non-decreasing property
            acc_sum = 0
            for v in range(nums[i] + 1):
                acc_sum = (acc_sum + dp[i-1][v]) % MOD
                dp[i][v] = acc_sum
        
        # Calculate the number of valid (arr1, arr2) pairs
        result = 0
        for arr1_end in range(nums[-1] + 1):
            result = (result + dp[n-1][arr1_end]) % MOD
        
        return result