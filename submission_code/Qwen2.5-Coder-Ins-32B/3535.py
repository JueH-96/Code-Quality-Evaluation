class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        max_val = max(nums)
        
        # dp[i][j] will store the number of ways to form a non-decreasing array of length i with last element j
        dp_inc = [[0] * (max_val + 1) for _ in range(n + 1)]
        dp_dec = [[0] * (max_val + 1) for _ in range(n + 1)]
        
        # Base case: there's one way to form an array of length 0
        for j in range(max_val + 1):
            dp_inc[0][j] = 1
            dp_dec[0][j] = 1
        
        # Fill dp_inc
        for i in range(1, n + 1):
            for j in range(max_val + 1):
                dp_inc[i][j] = dp_inc[i][j - 1]
                if j <= nums[i - 1]:
                    dp_inc[i][j] = (dp_inc[i][j] + dp_inc[i - 1][j]) % MOD
        
        # Fill dp_dec
        for i in range(1, n + 1):
            for j in range(max_val, -1, -1):
                dp_dec[i][j] = dp_dec[i][j + 1]
                if j <= nums[i - 1]:
                    dp_dec[i][j] = (dp_dec[i][j] + dp_dec[i - 1][j]) % MOD
        
        # Calculate the result
        result = 0
        for j in range(max_val + 1):
            if j <= nums[0]:
                result = (result + dp_inc[n][j] * dp_dec[n][nums[0] - j]) % MOD
        
        return result