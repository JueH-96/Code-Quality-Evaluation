class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # dp[i][j] will store the number of ways to form arr1 and arr2
        # such that arr1[i] + arr2[i] = nums[i] and arr1[i] <= j
        dp = [[0] * 51 for _ in range(n)]
        
        # Initialize the first element
        for j in range(nums[0] + 1):
            dp[0][j] = 1
        
        # Fill the dp table
        for i in range(1, n):
            for j in range(51):
                dp[i][j] = dp[i - 1][j]  # carry over the previous counts
                if j > 0:
                    dp[i][j] += dp[i][j - 1]  # add counts from the previous row
                    dp[i][j] %= MOD
                
                if j <= nums[i]:
                    dp[i][j] += dp[i - 1][nums[i] - j]  # add valid pairs
                    dp[i][j] %= MOD
        
        # The result is the sum of all valid pairs for the last element
        result = sum(dp[n - 1]) % MOD
        return result