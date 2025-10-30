import sys
from collections import deque

def main():
    H, W, T = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        line = sys.stdin.readline().strip()
        grid.append(list(line))
    
    # Collect S, G, and candies.
    S = None
    G = None
    points = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                S = (i, j)
            elif grid[i][j] == 'G':
                G = (i, j)
            elif grid[i][j] == 'o':
                points.append((i, j))
    
    # Create points list as [S, c1, c2, ..., G]
    if S is None or G is None:
        print(-1)
        return
    points = [S] + points + [G]
    m = len(points) - 2  # Number of candies
    
    # Precompute distances between all points.
    dist = [[float('inf')] * len(points) for _ in range(len(points))]
    for u_idx in range(len(points)):
        u = points[u_idx]
        i, j = u
        visited = [[-1 for _ in range(W)] for _ in range(H)]
        q = deque()
        q.append((i, j))
        visited[i][j] = 0
        while q:
            x, y = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    if grid[nx][ny] != '#' and visited[nx][ny] == -1:
                        visited[nx][ny] = visited[x][y] + 1
                        q.append((nx, ny))
        # Fill the distance matrix
        for v_idx in range(len(points)):
            v = points[v_idx]
            i_v, j_v = v
            if visited[i_v][j_v] == -1:
                dist[u_idx][v_idx] = float('inf')
            else:
                dist[u_idx][v_idx] = visited[i_v][j_v]
    
    INF_STEPS = 10**18
    max_mask = 1 << m
    dp = [[INF_STEPS for _ in range(len(points))] for _ in range(max_mask)]
    dp[0][0] = 0  # Starting at S with 0 candies collected
    
    all_masks = []
    for mask in range(max_mask):
        k = bin(mask).count('1')
        all_masks.append((mask, k))
    all_masks.sort(key=lambda x: x[1])
    
    max_candies = 0
    
    for mask, k in all_masks:
        for u_idx in range(len(points)):
            current_steps = dp[mask][u_idx]
            if current_steps == INF_STEPS:
                continue
            # Try moving to each unvisited candy
            for v in range(m):
                v_idx = 1 + v
                if (mask & (1 << v)) == 0:
                    new_mask = mask | (1 << v)
                    if dist[u_idx][v_idx] == float('inf'):
                        continue
                    new_steps = current_steps + dist[u_idx][v_idx]
                    if new_steps < dp[new_mask][v_idx]:
                        dp[new_mask][v_idx] = new_steps
            # Try moving directly to G
            g_idx = len(points) - 1
            if dist[u_idx][g_idx] == float('inf'):
                continue
            total_steps = current_steps + dist[u_idx][g_idx]
            if total_steps <= T:
                count = bin(mask).count('1')
                if count > max_candies:
                    max_candies = count
    
    if max_candies == 0:
        s_to_g_dist = dist[0][len(points)-1]
        if s_to_g_dist <= T:
            print(0)
        else:
            print(-1)
    else:
        print(max_candies)

if __name__ == '__main__':
    main()