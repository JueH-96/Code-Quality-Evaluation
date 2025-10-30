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
        
        INF = 10**9
        dp = [[INF] * m for _ in range(m)]
        
        for length in range(2, m+1, 2):
            for i in range(0, m - length + 1):
                j = i + length - 1
                best = INF
                for k in range(i, j, 2):
                    left_val = 0
                    if k > i:
                        left_val = dp[i][k-1]
                    right_val = 0
                    if k < j-1:
                        right_val = dp[k+1][j-1]
                    cost_pair = min(x, A[j] - A[k])
                    total = left_val + right_val + cost_pair
                    if total < best:
                        best = total
                dp[i][j] = best
        
        return dp[0][m-1]