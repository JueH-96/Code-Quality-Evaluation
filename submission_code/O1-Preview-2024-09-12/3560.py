class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        from collections import deque
        N = len(positions) + 1  # Number of positions including knight's initial position
        pos_list = [[kx, ky]] + positions  # positions[0] is knight's starting position
        pos_indices = {tuple(pos): idx for idx, pos in enumerate(pos_list)}
        max_x = max(pos[0] for pos in pos_list) + 5
        max_y = max(pos[1] for pos in pos_list) + 5
        max_x = min(max_x, 50)
        max_y = min(max_y, 50)
        
        # Precompute distances between all positions of interest using BFS
        distances = [[0]*N for _ in range(N)]
        moves = [(-2,-1), (-1,-2), (1,-2), (2,-1), 
                 (2,1), (1,2), (-1,2), (-2,1)]
        for i in range(N):
            # BFS from pos_list[i]
            queue = deque()
            visited = [[-1]*50 for _ in range(50)]
            x0, y0 = pos_list[i]
            queue.append((x0, y0))
            visited[x0][y0] = 0
            while queue:
                x, y = queue.popleft()
                d = visited[x][y]
                pos = (x, y)
                if pos in pos_indices:
                    j = pos_indices[pos]
                    distances[i][j] = d
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and visited[nx][ny] == -1:
                        visited[nx][ny] = d + 1
                        queue.append((nx, ny))
        from functools import lru_cache
        full_mask = (1 << (N-1)) - 1  # Masks for the pawns (excluding starting position)
        @lru_cache(None)
        def dp(mask, kp_idx, isAlice):
            if mask == 0:
                return 0
            if isAlice:
                max_total = float('-inf')
                for i in range(1, N):
                    if mask & (1 << (i -1)):
                        cost = distances[kp_idx][i]
                        next_mask = mask ^ (1 << (i -1))
                        total = cost + dp(next_mask, i, False)
                        if total > max_total:
                            max_total = total
                return max_total
            else:
                min_total = float('inf')
                for i in range(1, N):
                    if mask & (1 << (i -1)):
                        cost = distances[kp_idx][i]
                        next_mask = mask ^ (1 << (i -1))
                        total = cost + dp(next_mask, i, True)
                        if total < min_total:
                            min_total = total
                return min_total
        result = dp(full_mask, 0, True)
        return result