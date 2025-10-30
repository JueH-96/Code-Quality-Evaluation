class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        dp = [[0]*n for _ in range(m)]
        dp2 = [[0]*n for _ in range(m)]
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                if i+1<m and j+1<n:
                    dp[i][j] = max(dp[i+1][j], dp[i][j+1]) + board[i][j]
                else:
                    dp[i][j] = board[i][j]
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                if i+1<m and j+1<n:
                    dp2[i][j] = max(dp2[i+1][j], dp2[i][j+1]) + (dp[i][j] if dp[i][j]>0 else 0)
                else:
                    dp2[i][j] = dp[i][j]
        return max(dp2[i][j] for i in range(m) for j in range(n))