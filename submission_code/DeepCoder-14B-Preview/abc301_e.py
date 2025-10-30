from collections import deque

def bfs(grid, H, W, start_i, start_j):
    visited = [[-1 for _ in range(W)] for __ in range(H)]
    q = deque()
    q.append((start_i, start_j))
    visited[start_i][start_j] = 0
    while q:
        x, y = q.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < H and 0 <= ny < W:
                if grid[nx][ny] != '#' and visited[nx][ny] == -1:
                    visited[nx][ny] = visited[x][y] + 1
                    q.append((nx, ny))
    return visited

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    H = int(input[idx]); idx +=1
    W = int(input[idx]); idx +=1
    T = int(input[idx]); idx +=1
    grid = []
    for _ in range(H):
        grid.append(input[idx])
        idx +=1
    
    start = None
    goal = None
    candies = []
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            if c == 'S':
                start = (i, j)
            elif c == 'G':
                goal = (i, j)
            elif c == 'o':
                candies.append((i, j))
    
    n = len(candies)
    if n == 0:
        # Compute distance from start to goal
        if start is None or goal is None:
            print(-1)
            return
        visited = bfs(grid, H, W, start[0], start[1])
        distance = visited[goal[0]][goal[1]] if visited[goal[0]][goal[1]] != -1 else float('inf')
        if distance <= T:
            print(0)
        else:
            print(-1)
        return
    
    # Create points list: [start, candy1, candy2, ..., goal]
    points = [start] + candies + [goal]
    m = len(points)
    INF = float('inf')
    dist = [[INF] * m for _ in range(m)]
    
    for i in range(m):
        start_i, start_j = points[i]
        visited = bfs(grid, H, W, start_i, start_j)
        for j in range(m):
            end_i, end_j = points[j]
            if visited[end_i][end_j] == -1:
                dist[i][j] = INF
            else:
                dist[i][j] = visited[end_i][end_j]
    
    # Initialize DP
    max_mask = 1 << n
    dp = [[INF] * m for _ in range(max_mask)]
    dp[0][0] = 0  # Starting at S with no candies collected
    
    for mask in range(max_mask):
        for u in range(m):
            if u == m - 1:  # Skip if current position is goal
                continue
            if dp[mask][u] == INF:
                continue
            # Try to collect each possible candy
            for v in range(1, n + 1):  # v is the index in points (1..n are candies)
                if (mask & (1 << (v - 1))) != 0:
                    continue  # Already collected
                d = dist[u][v]
                if d == INF:
                    continue
                new_mask = mask | (1 << (v - 1))
                new_cost = dp[mask][u] + d
                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
    
    max_candies = -1
    for mask in range(max_mask):
        for u in range(m):
            if u == m - 1:
                continue
            if dp[mask][u] == INF:
                continue
            # Distance from current position to goal
            d_to_goal = dist[u][m - 1]
            if d_to_goal == INF:
                continue
            total = dp[mask][u] + d_to_goal
            if total <= T:
                count = bin(mask).count('1')
                if count > max_candies:
                    max_candies = count
    
    if max_candies == -1:
        print(-1)
    else:
        print(max_candies)

if __name__ == '__main__':
    main()