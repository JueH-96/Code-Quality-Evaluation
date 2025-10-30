class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Initialize DP table
        dp = [[0] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        # Fill DP table
        for num in nums:
            for t in range(n, 0, -1):
                for s in range(k, num - 1, -1):
                    dp[t][s] = (dp[t][s] + dp[t-1][s - num]) % MOD
        
        # Precompute powers of 2
        power_of_two = [pow(2, n - t, MOD) for t in range(n + 1)]
        
        # Calculate the final answer
        answer = 0
        for t in range(n + 1):
            answer = (answer + dp[t][k] * power_of_two[t]) % MOD
        
        return answer