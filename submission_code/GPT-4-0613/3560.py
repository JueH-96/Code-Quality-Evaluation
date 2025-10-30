class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        dp = [0] * (1 << len(positions))
        dist = [[0] * len(positions) for _ in range(len(positions))]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist[i][j] = dist[j][i] = self.distance(positions[i], positions[j])
        for mask in range(1, len(dp)):
            b = [i for i in range(len(positions)) if mask & (1 << i)]
            for i in b:
                dp[mask] = max(dp[mask], (dp[mask ^ (1 << i)] if len(b) > 1 else 0) + dist[i][b[0]] if i != b[0] else 0)
        return max((self.distance([kx, ky], positions[i]) + dp[(1 << len(positions)) - 1 - (1 << i)]) for i in range(len(positions)))

    def distance(self, a, b):
        x, y, X, Y = a[0], a[1], b[0], b[1]
        d = [[0] * 50 for _ in range(50)]
        d[x][y] = 1
        for _ in range(50 * 50):
            for i in range(50):
                for j in range(50):
                    if d[i][j]:
                        for dx, dy in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
                            if 0 <= i + dx < 50 and 0 <= j + dy < 50:
                                if not d[i + dx][j + dy]:
                                    d[i + dx][j + dy] = d[i][j] + 1
        return d[X][Y] - 1