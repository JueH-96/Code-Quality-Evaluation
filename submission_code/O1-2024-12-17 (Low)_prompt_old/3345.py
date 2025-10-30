class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        import sys
        MOD = 10**9 + 7
        
        n = len(nums)
        
        # Precompute powers of 2 up to n
        pow2 = [1]*(n+1)
        for i in range(1,n+1):
            pow2[i] = (pow2[i-1]*2) % MOD
        
        # dp[r][s] will store how many subsets of size r have sum s
        dp = [[0]*(k+1) for _ in range(n+1)]
        dp[0][0] = 1  # There's exactly 1 subset of size 0 with sum 0: the empty subset
        
        # Build up dp for each number in nums
        for val in nums:
            # Update dp in descending order of r and s to avoid using updated values prematurely
            for r in range(n, 0, -1):
                for s in range(k, val-1, -1):
                    dp[r][s] = (dp[r][s] + dp[r-1][s-val]) % MOD
        
        # Now sum up dp[r][k]*2^(n-r) for r in [0..n]
        # dp[r][k] is how many subsets of size r sum to k
        # Each such subset S is contained in exactly 2^(n-r) supersets.
        ans = 0
        for r in range(n+1):
            ans = (ans + dp[r][k]*pow2[n-r]) % MOD
        
        return ans