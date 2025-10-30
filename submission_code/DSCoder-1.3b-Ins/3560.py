class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        positions.append([kx, ky])
        positions.sort()
        n = len(positions)
        dp = [0]*n
        dp[0] = 1
        res = 0
        for i in range(1, n):
            dp[i] = 1
            for j in range(i):
                if positions[i][1] < positions[j][1]:
                    dp[i] = max(dp[i], dp[j]+1)
            res = max(res, dp[i])
        return res-1