import sys
from collections import deque

def main():
    H, W, T = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]

    # Find S, G, and candies
    S_pos = None
    G_pos = None
    candies = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                S_pos = (i, j)
            elif grid[i][j] == 'G':
                G_pos = (i, j)
            elif grid[i][j] == 'o':
                candies.append((i, j))
    m = len(candies)
    K = m + 2  # Number of key nodes: S, m candies, G

    # Build key_nodes list
    key_nodes = [S_pos] + candies + [G_pos]

    # Precompute distances between all pairs of key nodes
    INF = float('inf')
    distances = [[INF] * K for _ in range(K)]

    def bfs(sx, sy):
        dist = [[INF] * W for _ in range(H)]
        q = deque()
        dist[sx][sy] = 0
        q.append((sx, sy))
        while q:
            x, y = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                    if dist[nx][ny] == INF:
                        dist[nx][ny] = dist[x][y] + 1
                        q.append((nx, ny))
        return dist

    for i in range(K):
        sx, sy = key_nodes[i]
        grid_dist = bfs(sx, sy)
        for j in range(K):
            x, y = key_nodes[j]
            distances[i][j] = grid_dist[x][y]

    # Check if S can reach G
    if distances[0][K-1] > T:
        print(-1)
        return

    # Dynamic Programming setup
    dp = [[INF] * (m + 1) for _ in range(1 << m)]
    dp[0][0] = 0  # Start at S with no candies collected

    for mask in range(1 << m):
        for u in range(m + 1):  # u can be S (0) or any candy (1..m)
            if dp[mask][u] == INF:
                continue
            # Try to transition to all candies not in mask
            for j in range(m):
                if not (mask & (1 << j)):
                    v_node = j + 1  # key node index for candy j
                    if distances[u][v_node] == INF:
                        continue
                    new_mask = mask | (1 << j)
                    new_cost = dp[mask][u] + distances[u][v_node]
                    if new_cost < dp[new_mask][v_node]:
                        dp[new_mask][v_node] = new_cost

    max_candies = -1
    G_index = K - 1
    for mask in range(1 << m):
        for u in range(m + 1):
            if distances[u][G_index] == INF:
                continue
            total_steps = dp[mask][u] + distances[u][G_index]
            if total_steps <= T:
                cnt = bin(mask).count('1')
                if cnt > max_candies:
                    max_candies = cnt

    print(max_candies if max_candies != -1 else -1)

if __name__ == "__main__":
    main()