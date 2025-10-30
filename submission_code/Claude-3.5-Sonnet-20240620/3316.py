class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()
        
        # dp[i][j] represents the sum of powers for subsequences ending at index i with length j
        dp = [[0] * (k + 1) for _ in range(n)]
        
        # Initialize for subsequences of length 1
        for i in range(n):
            dp[i][1] = 1
        
        for length in range(2, k + 1):
            for i in range(length - 1, n):
                for j in range(i):
                    power = abs(nums[i] - nums[j])
                    dp[i][length] = (dp[i][length] + dp[j][length - 1]) % MOD
        
        return sum(dp[i][k] for i in range(k - 1, n)) % MOD