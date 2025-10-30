from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        # Precompute "pattern" for each possible subarray length m=1...n.
        # pattern[m] is a list of offsets delta (0-indexed) such that C(m-1, delta) mod 2 == 1.
        # Using Lucas theorem: C(n, k) mod 2 == 1 if and only if (k & ~n) == 0.
        pattern = [[] for _ in range(n+1)]
        for m in range(1, n+1):
            # m-1 is our "n" in binom coefficients.
            top = m - 1
            # For delta from 0 to top:
            for delta in range(m):
                # Check if every 1-bit in delta is also present in top.
                if delta & ~top == 0:
                    pattern[m].append(delta)
                    
        # Precompute the XOR-score for every subarray f(i, j) using the pattern.
        # f[i][j] will hold the XOR-score for subarray nums[i..j]
        f = [[0]*n for _ in range(n)]
        for i in range(n):
            # Subarray of length 1: just the element itself.
            f[i][i] = nums[i]
            # For further j
            for j in range(i+1, n):
                length = j - i + 1  # subarray length
                # Compute XOR over indices: for each delta in pattern[length] add nums[i+delta]
                xor_val = 0
                for delta in pattern[length]:
                    xor_val ^= nums[i+delta]
                f[i][j] = xor_val
        
        # Precompute dp table: dp[i][j] = maximum of f(a,b) for i<=a<=b<=j.
        dp = [[0]*n for _ in range(n)]
        # Fill dp diagonally.
        for i in range(n):
            dp[i][i] = f[i][i]
        # for subarray lengths L from 2 to n:
        for L in range(2, n+1):
            for i in range(n - L + 1):
                j = i + L - 1
                # dp[i][j] is maximum among:
                #  - f(i,j) computed for the subarray from i to j,
                #  - dp[i+1][j] (best in subarray starting at i+1),
                #  - dp[i][j-1] (best in subarray ending at j)
                dp[i][j] = max(f[i][j], dp[i+1][j], dp[i][j-1])
                
        # Answer the queries.
        res = []
        for l, r in queries:
            res.append(dp[l][r])
        return res