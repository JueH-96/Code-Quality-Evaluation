from collections import deque

def solve():
    H, W, T = map(int, input().split())
    grid = []
    for _ in range(H):
        grid.append(input().strip())
    
    # Find start, goal, and candy positions
    start = None
    goal = None
    candies = []
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'G':
                goal = (i, j)
            elif grid[i][j] == 'o':
                candies.append((i, j))
    
    # BFS to find shortest distances
    def bfs(start_pos):
        dist = [[-1] * W for _ in range(H)]
        queue = deque([start_pos])
        dist[start_pos[0]][start_pos[1]] = 0
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#' and dist[nr][nc] == -1:
                    dist[nr][nc] = dist[r][c] + 1
                    queue.append((nr, nc))
        
        return dist
    
    # Get distance from start to all positions
    start_dist = bfs(start)
    
    # Check if goal is reachable
    if start_dist[goal[0]][goal[1]] == -1 or start_dist[goal[0]][goal[1]] > T:
        print(-1)
        return
    
    # If no candies, just check if we can reach goal
    if not candies:
        print(0)
        return
    
    n = len(candies)
    
    # Compute distances between all important points
    # 0: start, 1: goal, 2 to n+1: candies
    points = [start, goal] + candies
    dist_matrix = [[float('inf')] * (n + 2) for _ in range(n + 2)]
    
    for i in range(n + 2):
        dist_from_i = bfs(points[i])
        for j in range(n + 2):
            if dist_from_i[points[j][0]][points[j][1]] != -1:
                dist_matrix[i][j] = dist_from_i[points[j][0]][points[j][1]]
    
    # DP with bitmask
    # dp[mask][i] = minimum moves to visit candies in mask and end at position i
    # i: 0 = start, 1 = goal, 2 to n+1 = candy positions
    INF = float('inf')
    dp = [[INF] * (n + 2) for _ in range(1 << n)]
    
    # Start at position 0 (start square)
    dp[0][0] = 0
    
    max_candies = 0
    
    for mask in range(1 << n):
        for pos in range(n + 2):
            if dp[mask][pos] == INF:
                continue
            
            # Try to move to goal
            if pos != 1 and dp[mask][pos] + dist_matrix[pos][1] <= T:
                candy_count = bin(mask).count('1')
                max_candies = max(max_candies, candy_count)
            
            # Try to visit unvisited candies
            for candy_idx in range(n):
                if mask & (1 << candy_idx):
                    continue
                
                candy_pos = candy_idx + 2
                new_mask = mask | (1 << candy_idx)
                new_dist = dp[mask][pos] + dist_matrix[pos][candy_pos]
                
                if new_dist <= T:
                    dp[new_mask][candy_pos] = min(dp[new_mask][candy_pos], new_dist)
    
    print(max_candies)

solve()