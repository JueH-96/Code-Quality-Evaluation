class Solution:
    def minChanges(self, s: str) -> int:
        N = len(s)
        dp = [[[float('inf')] * 2 for _ in range(2)] for _ in range(N+1)]
        dp[0][0][0] = dp[0][1][0] = 0
        for i in range(N):
            for c in [0, 1]:
                for l in [0, 1]:
                    if dp[i][c][l] < float('inf'):
                        # Continue current block
                        cost = 0 if int(s[i]) == c else 1
                        c_new = c
                        l_new = (l + 1) % 2
                        dp[i+1][c_new][l_new] = min(dp[i+1][c_new][l_new], dp[i][c][l] + cost)
                        # Start new block if current block length is even
                        if l == 0:
                            for c_new in [0, 1]:
                                cost = 0 if int(s[i]) == c_new else 1
                                l_new = 1
                                dp[i+1][c_new][l_new] = min(dp[i+1][c_new][l_new], dp[i][c][l] + cost)
        return min(dp[N][0][0], dp[N][1][0])