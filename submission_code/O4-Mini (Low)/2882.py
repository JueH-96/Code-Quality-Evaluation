class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        MOD = 10**9 + 7
        # Precompute all i^x such that i^x <= n
        powers = []
        i = 1
        while True:
            p = i ** x
            if p > n:
                break
            powers.append(p)
            i += 1
        
        # dp[s] = number of ways to get sum s using unique powers
        dp = [0] * (n + 1)
        dp[0] = 1
        
        # For each power, update dp in reverse to ensure uniqueness
        for p in powers:
            for s in range(n, p - 1, -1):
                dp[s] = (dp[s] + dp[s - p]) % MOD
        
        return dp[n]


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.numberOfWays(10, 2))  # Output: 1
    # Example 2
    print(sol.numberOfWays(4, 1))   # Output: 2