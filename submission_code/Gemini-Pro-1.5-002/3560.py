class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        n = len(positions)
        dist = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                q = [(positions[i][0], positions[i][1], 0)]
                visited = set([(positions[i][0], positions[i][1])])
                while q:
                    x, y, d = q.pop(0)
                    if x == positions[j][0] and y == positions[j][1]:
                        dist[i][j] = d
                        break
                    moves = [(x + 1, y + 2), (x + 1, y - 2), (x - 1, y + 2), (x - 1, y - 2),
                             (x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1)]
                    for nx, ny in moves:
                        if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            q.append((nx, ny, d + 1))

        start_dist = [0] * n
        for i in range(n):
            q = [(kx, ky, 0)]
            visited = set([(kx, ky)])
            while q:
                x, y, d = q.pop(0)
                if x == positions[i][0] and y == positions[i][1]:
                    start_dist[i] = d
                    break
                moves = [(x + 1, y + 2), (x + 1, y - 2), (x - 1, y + 2), (x - 1, y - 2),
                         (x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1)]
                for nx, ny in moves:
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny, d + 1))
        
        dp = {}
        def solve(mask, turn):
            if (mask, turn) in dp:
                return dp[(mask, turn)]
            
            if mask == (1 << n) - 1:
                return 0
            
            res = 0 if turn == 0 else float('inf')
            for i in range(n):
                if (mask >> i) & 1 == 0:
                    new_mask = mask | (1 << i)
                    val = start_dist[i] if mask == 0 else 0
                    
                    temp = solve(new_mask, 1 - turn)
                    if turn == 0:
                        
                        res = max(res, val + temp)
                    else:
                        res = min(res, temp)
            dp[(mask, turn)] = res
            return res
        
        return solve(0, 0)