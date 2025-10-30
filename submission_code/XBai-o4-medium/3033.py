class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        diffs = []
        for i in range(n):
            if s1[i] != s2[i]:
                diffs.append(i)
        if len(diffs) % 2 != 0:
            return -1
        m = len(diffs)
        if m == 0:
            return 0
        # Initialize DP table
        dp = [[0] * m for _ in range(m)]
        # Fill the DP table
        for L in range(2, m + 1, 2):
            for i in range(m - L + 1):
                j = i + L - 1
                if L == 2:
                    dp[i][j] = min(x, diffs[j] - diffs[i])
                else:
                    dp[i][j] = float('inf')
                    # Try all possible positions to pair with i
                    for m_pos in range(i + 1, j + 1, 2):
                        cost_pair = min(x, diffs[m_pos] - diffs[i])
                        # Compute left part: i+1 to m_pos-1
                        left = dp[i + 1][m_pos - 1] if (i + 1 <= m_pos - 1) else 0
                        # Compute right part: m_pos+1 to j
                        right = dp[m_pos + 1][j] if (m_pos + 1 <= j) else 0
                        total = cost_pair + left + right
                        if total < dp[i][j]:
                            dp[i][j] = total
        return dp[0][m - 1]