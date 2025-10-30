class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Preprocess nodes: initial position plus all pawn positions
        nodes = [(kx, ky)] + positions
        m = len(positions)
        n_nodes = m + 1
        
        # Precompute distance matrix using BFS for each node
        dist = [[0] * n_nodes for _ in range(n_nodes)]
        from collections import deque
        
        for i in range(n_nodes):
            x, y = nodes[i]
            visited = [[-1] * 50 for _ in range(50)]
            q = deque()
            q.append((x, y))
            visited[x][y] = 0
            while q:
                cx, cy = q.popleft()
                for dx, dy in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                               (1, -2), (1, 2), (2, -1), (2, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and visited[nx][ny] == -1:
                        visited[nx][ny] = visited[cx][cy] + 1
                        q.append((nx, ny))
            # Fill the distance matrix for this node
            for j in range(n_nodes):
                dx, dy = nodes[j]
                dist[i][j] = visited[dx][dy]
        
        # Define the recursive DFS function with memoization
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dfs(current_node, mask):
            if mask == 0:
                return 0
            pawns_taken = m - bin(mask).count('1')
            if pawns_taken % 2 == 0:
                # Alice's turn: maximize the result
                max_sum = -float('inf')
                for p in range(m):
                    if (mask & (1 << p)):
                        moves = dist[current_node][p + 1]
                        new_mask = mask ^ (1 << p)
                        total = moves + dfs(p + 1, new_mask)
                        if total > max_sum:
                            max_sum = total
                return max_sum if max_sum != -float('inf') else 0
            else:
                # Bob's turn: minimize the result
                min_sum = float('inf')
                for p in range(m):
                    if (mask & (1 << p)):
                        moves = dist[current_node][p + 1]
                        new_mask = mask ^ (1 << p)
                        total = moves + dfs(p + 1, new_mask)
                        if total < min_sum:
                            min_sum = total
                return min_sum if min_sum != float('inf') else 0
        
        initial_mask = (1 << m) - 1
        return dfs(0, initial_mask)