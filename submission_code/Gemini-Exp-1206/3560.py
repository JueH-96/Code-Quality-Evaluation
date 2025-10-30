class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        def bfs(sx, sy, ex, ey):
            q = [(sx, sy, 0)]
            visited = {(sx, sy)}
            while q:
                x, y, dist = q.pop(0)
                if x == ex and y == ey:
                    return dist
                for dx, dy in [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny, dist + 1))
            return -1

        n = len(positions)
        dist = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                dist[i][j] = bfs(positions[i][0], positions[i][1], positions[j][0], positions[j][1])

        knight_dist = [0] * n
        for i in range(n):
            knight_dist[i] = bfs(kx, ky, positions[i][0], positions[i][1])

        dp = {}

        def solve(kx, ky, mask, turn):
            if mask == (1 << n) - 1:
                return 0
            if (kx, ky, mask, turn) in dp:
                return dp[(kx, ky, mask, turn)]

            if turn == 0:  # Alice's turn
                res = 0
                for i in range(n):
                    if (mask >> i) & 1 == 0:
                        res = max(res, knight_dist[i] + solve(positions[i][0], positions[i][1], mask | (1 << i), 1))
            else:  # Bob's turn
                res = float('inf')
                for i in range(n):
                    if (mask >> i) & 1 == 0:
                        res = min(res, knight_dist[i] + solve(positions[i][0], positions[i][1], mask | (1 << i), 0))
            
            dp[(kx, ky, mask, turn)] = res
            return res
        
        
        
        dp2 = {}
        def solve2(kx, ky, mask, turn):
            if mask == (1 << n) - 1:
                return 0
            if (kx, ky, mask, turn) in dp2:
                return dp2[(kx, ky, mask, turn)]
            
            
            
            if turn == 0:
                res = 0
                for i in range(n):
                    if (mask >> i) & 1 == 0:
                        d = bfs(kx, ky, positions[i][0], positions[i][1])
                        res = max(res, d + solve2(positions[i][0], positions[i][1], mask | (1 << i), 1))
            else:
                res = float('inf')
                for i in range(n):
                    if (mask >> i) & 1 == 0:
                        d = bfs(kx, ky, positions[i][0], positions[i][1])
                        res = min(res, d + solve2(positions[i][0], positions[i][1], mask | (1 << i), 0))
            dp2[(kx, ky, mask, turn)] = res
            return res

        return solve2(kx, ky, 0, 0)