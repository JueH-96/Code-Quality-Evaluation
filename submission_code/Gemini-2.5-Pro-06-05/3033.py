class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        """
        Calculates the minimum cost to make s1 equal to s2 using the given operations.

        The method uses dynamic programming on the indices where s1 and s2 differ.
        1.  First, it identifies all indices where s1[i] != s2[i]. If the number of
            such indices is odd, it's impossible to make the strings equal, so -1
            is returned.
        2.  The problem is equivalent to finding a minimum cost perfect matching
            on these differing indices. The cost to match two indices i and j is
            min(x, j - i).
        3.  A dynamic programming approach is used to find this minimum cost.
            Let dp[i][j] be the minimum cost to resolve the differences from
            index diffs[i] to diffs[j].
        4.  The state transition for dp[i][j] considers two main cases for the
            first difference diffs[i]:
            a) Pair diffs[i] with its adjacent difference diffs[i+1]. The cost is
               the cost for this pair plus the optimal cost for the rest (dp[i+2][j]).
            b) Pair diffs[i] with the last difference in the range, diffs[j]. The
               cost is the cost for this pair plus the optimal cost for the inner
               part (dp[i+1][j-1]).
        5.  The DP table is filled for increasing lengths of subarrays of diffs,
            and the final answer is the value for the entire range of differences.
        """
        diffs = [i for i, (c1, c2) in enumerate(zip(s1, s2)) if c1 != c2]
        
        m = len(diffs)
        if m % 2 != 0:
            return -1
        
        if m == 0:
            return 0
        
        # dp[i][j] = min cost to resolve differences from diffs[i] to diffs[j]
        dp = [[0] * m for _ in range(m)]

        for length in range(2, m + 1, 2):
            for i in range(m - length + 1):
                j = i + length - 1
                
                # Option 1: Pair diffs[i] with diffs[i+1]
                cost1 = (diffs[i+1] - diffs[i])
                if length > 2:
                    cost1 += dp[i+2][j]
                
                # Option 2: Pair diffs[i] with diffs[j]
                cost2 = x
                if length > 2:
                    cost2 += dp[i+1][j-1]
                
                # We can choose the cheaper of the two strategies
                dp[i][j] = min(cost1, cost2)

        return dp[0][m-1]