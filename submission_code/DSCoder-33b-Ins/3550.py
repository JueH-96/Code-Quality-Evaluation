class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        dp = [[0]*n for _ in range(m)]
        dp[0][0] = board[0][0]
        for i in range(1, m):
            dp[i][0] = max(dp[i-1][0], board[i][0])
        for j in range(1, n):
            dp[0][j] = max(dp[0][j-1], board[0][j])
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]) + board[i][j]
        return dp[-1][-1]