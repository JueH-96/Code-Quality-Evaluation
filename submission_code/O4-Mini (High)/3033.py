class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        # Collect mismatch positions
        pos = [i for i, (a, b) in enumerate(zip(s1, s2)) if a != b]
        m = len(pos)
        # If odd number of mismatches, impossible
        if m & 1:
            return -1
        # No mismatches
        if m == 0:
            return 0
        # If x == 1, every pair (no matter how far) costs 1 by using one type-1 op
        if x == 1:
            return m // 2
        # If x is at least as large as the maximum distance between mismatches,
        # then it's never better to use the global flip (cost x). 
        # We can only use adjacent flips, and on a line metric the optimal
        # perfect matching is to pair neighbors in the sorted list.
        if x >= pos[-1] - pos[0]:
            res = 0
            for i in range(0, m, 2):
                res += pos[i+1] - pos[i]
            return res
        
        # Otherwise we do a classical DP for non-crossing perfect matching
        # on the sorted mismatch positions, with weight w(i,j) = min(x, pos[j]-pos[i]).
        INF = 10**18
        dp = [[INF] * m for _ in range(m)]
        
        # Base case: segments of length 2
        for i in range(m - 1):
            d = pos[i+1] - pos[i]
            dp[i][i+1] = d if d < x else x
        
        # Build up for even lengths 4,6,... up to m
        for length in range(4, m+1, 2):
            # segment starts at i, ends at j = i+length-1
            for i in range(0, m - length + 1):
                j = i + length - 1
                # Option 1: match endpoints i,j
                d = pos[j] - pos[i]
                cost_ij = d if d < x else x
                best = dp[i+1][j-1] + cost_ij
                # Option 2: split the segment into two even parts
                # k runs over possible split points so that both subsegments are even-length
                # (k-i+1) must be even => (k-i) odd => start from i+1, step by 2
                for k in range(i+1, j, 2):
                    tmp = dp[i][k] + dp[k+1][j]
                    if tmp < best:
                        best = tmp
                dp[i][j] = best
        
        return dp[0][m-1]