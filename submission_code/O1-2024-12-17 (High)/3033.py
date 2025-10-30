class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        
        # Build the list of mismatch positions
        mismatch_positions = []
        for i in range(n):
            if s1[i] != s2[i]:
                mismatch_positions.append(i)
        
        m = len(mismatch_positions)
        
        # If no mismatches, cost is 0
        if m == 0:
            return 0
        
        # If odd number of mismatches, impossible (since every flip toggles two bits)
        if m % 2 == 1:
            return -1
        
        # If flipping any two bits (operation #1) costs 1, then every pair costs 1
        # So we just need m/2 such flips
        if x == 1:
            return m // 2
        
        # Otherwise, let p = mismatch_positions
        p = mismatch_positions
        
        # If x >= the maximum distance between any two mismatches, then cost(i, j) = (p[j] - p[i])
        # for all i < j (no benefit from capping the distance at x).
        # In that case, the best way to minimize sum of distances on a line is to pair in
        # sorted order consecutively: (p[0],p[1]), (p[2],p[3])...
        max_dist = p[-1] - p[0]  # largest possible distance among mismatch positions
        if x >= max_dist:
            total = 0
            for i in range(0, m, 2):
                total += p[i+1] - p[i]
            return total
        
        # Otherwise, we do a DP over all ways to pair mismatches.
        # Cost function c(i, j) = min(x, p[j] - p[i])
        
        # Precompute the cost for pairing mismatch i with j
        c = [[0]*m for _ in range(m)]
        for i in range(m):
            for j in range(i+1, m):
                c[i][j] = min(x, p[j] - p[i])
        
        # dp[i][j] = minimum cost to fix all mismatches from index i to j in the mismatch_positions array
        # We'll fill dp for intervals of even length only.
        INF = 10**9
        dp = [[0]*m for _ in range(m)]
        
        # length goes in steps of 2 because we can only fix an even number of mismatches in a segment
        for length in range(2, m+1, 2):
            for i in range(m - length + 1):
                j = i + length - 1
                dp[i][j] = INF
                # We pair p[i] with p[k], and fix subproblems [i+1..k-1] and [k+1..j]
                # k must have the same parity difference so that subproblems have even length
                for k in range(i+1, j+1, 2):
                    cost_val = dp[i+1][k-1] + c[i][k] + dp[k+1][j]
                    if cost_val < dp[i][j]:
                        dp[i][j] = cost_val
        
        return dp[0][m-1]