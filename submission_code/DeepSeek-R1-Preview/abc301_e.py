import sys
import heapq
from collections import deque

def main():
    H, W, T = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        grid.append(list(sys.stdin.readline().strip()))
    
    # Find S, G, and candies
    S = None
    G = None
    candies = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                S = (i, j)
            elif grid[i][j] == 'G':
                G = (i, j)
            elif grid[i][j] == 'o':
                candies.append((i, j))
    
    # Collect points: S, candies, G
    points = [S] + candies + [G]
    C = len(candies)
    num_points = len(points)
    G_index = num_points - 1

    # Precompute distances between all points using BFS
    dist = [[float('inf')] * num_points for _ in range(num_points)]
    for u in range(num_points):
        start_x, start_y = points[u]
        visited = [[-1 for _ in range(W)] for _ in range(H)]
        q = deque()
        q.append((start_x, start_y))
        visited[start_x][start_y] = 0
        while q:
            x, y = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    if grid[nx][ny] != '#' and visited[nx][ny] == -1:
                        visited[nx][ny] = visited[x][y] + 1
                        q.append((nx, ny))
        # Update the distance matrix
        for v in range(num_points):
            xv, yv = points[v]
            if visited[xv][yv] != -1:
                dist[u][v] = visited[xv][yv]
            else:
                dist[u][v] = float('inf')

    # Initialize DP table
    INF = float('inf')
    dp = [[INF] * num_points for _ in range(1 << C)]
    dp[0][0] = 0

    # Priority queue: (steps, mask, u)
    heap = []
    heapq.heappush(heap, (0, 0, 0))

    while heap:
        steps, mask, u = heapq.heappop(heap)
        if steps > dp[mask][u]:
            continue
        # If current position is G, skip further transitions
        if u == G_index:
            continue
        for v in range(num_points):
            if v == u:
                continue
            d = dist[u][v]
            if d == float('inf'):
                continue
            new_steps = steps + d
            if new_steps > T:
                continue
            # Determine new_mask
            if v == 0 or v == G_index:
                new_mask = mask
            else:
                # v is a candy
                candy_index = v - 1
                if (mask & (1 << candy_index)) == 0:
                    new_mask = mask | (1 << candy_index)
                else:
                    new_mask = mask
            if new_steps < dp[new_mask][v]:
                dp[new_mask][v] = new_steps
                heapq.heappush(heap, (new_steps, new_mask, v))

    # Check all possible masks ending at G
    max_candies = -1
    for mask in range(1 << C):
        if dp[mask][G_index] <= T:
            count = bin(mask).count('1')
            if count > max_candies:
                max_candies = count

    print(max_candies if max_candies != -1 else -1)

if __name__ == "__main__":
    main()