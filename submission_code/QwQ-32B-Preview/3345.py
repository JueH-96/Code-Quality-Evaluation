class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        n = len(nums)
        
        # Initialize dp[j][m]
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1
        
        # Populate the dp table
        for num in nums:
            for j in range(k, num - 1, -1):
                for m in range(1, n + 1):
                    if j - num >= 0 and m - 1 >= 0:
                        dp[j][m] = (dp[j][m] + dp[j - num][m - 1]) % mod
        
        # Precompute 2^(n - m) for m from 0 to n
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow2[i] = (pow2[i - 1] * 2) % mod
        
        # Calculate total sum
        total_sum = 0
        for m in range(1, n + 1):
            if dp[k][m] > 0:
                total_sum = (total_sum + dp[k][m] * pow2[n - m]) % mod
        
        return total_sum