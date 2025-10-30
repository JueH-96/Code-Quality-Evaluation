class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        A = []
        for i in range(n):
            if s1[i] != s2[i]:
                A.append(i)
        m = len(A)
        if m % 2 != 0:
            return -1
        if m == 0:
            return 0
        
        dp = [[0] * m for _ in range(m)]
        
        for length in range(2, m + 1, 2):
            for l in range(0, m - length + 1):
                r = l + length - 1
                dp[l][r] = float('inf')
                for k in range(l + 1, r + 1, 2):
                    cost_pair = min(A[k] - A[l], x)
                    mid_segment = dp[l + 1][k - 1] if l + 1 <= k - 1 else 0
                    right_segment = dp[k + 1][r] if k + 1 <= r else 0
                    total = cost_pair + mid_segment + right_segment
                    if total < dp[l][r]:
                        dp[l][r] = total
        return dp[0][m - 1]