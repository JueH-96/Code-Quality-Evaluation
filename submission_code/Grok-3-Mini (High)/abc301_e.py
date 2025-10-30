import sys
from collections import deque

def shortest_path(grid, start, end, forbidden_set):
    H = len(grid)
    W = len(grid[0])
    dist_map = [[-1 for _ in range(W)] for _ in range(H)]
    sx, sy = start
    ex, ey = end
    dist_map[sx][sy] = 0
    queue = deque([(sx, sy)])
    while queue:
        cx, cy = queue.popleft()
        d = dist_map[cx][cy]
        if cx == ex and cy == ey:
            return d
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and (nx, ny) not in forbidden_set and dist_map[nx][ny] == -1:
                dist_map[nx][ny] = d + 1
                queue.append((nx, ny))
    return 10**9

def bfs_dist(grid, start, end):
    H = len(grid)
    W = len(grid[0])
    dist_map = [[-1 for _ in range(W)] for _ in range(H)]
    sx, sy = start
    ex, ey = end
    dist_map[sx][sy] = 0
    queue = deque([(sx, sy)])
    while queue:
        cx, cy = queue.popleft()
        d = dist_map[cx][cy]
        if cx == ex and cy == ey:
            return d
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and dist_map[nx][ny] == -1:
                dist_map[nx][ny] = d + 1
                queue.append((nx, ny))
    return -1

data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
T = int(data[index])
index += 1
grid = []
for i in range(H):
    row = data[index + i]
    grid.append(row)

# Find S, G, candy positions
S_pos = None
G_pos = None
candy_pos = []
for i in range(H):
    for j in range(W):
        cell = grid[i][j]
        if cell == 'S':
            S_pos = (i, j)
        elif cell == 'G':
            G_pos = (i, j)
        elif cell == 'o':
            candy_pos.append((i, j))

# Check if can reach G from S within T moves
dist_sg = bfs_dist(grid, S_pos, G_pos)
if dist_sg == -1 or dist_sg > T:
    print(-1)
    sys.exit()

# Proceed with key nodes
num_candies = len(candy_pos)
N = num_candies + 2
key_nodes = [S_pos] + candy_pos + [G_pos]
S_idx = 0
G_idx = N - 1
all_keys_set = set(key_nodes)

# Compute dist_matrix, symmetric
dist_matrix = [[0 for _ in range(N)] for _ in range(N)]
for i in range(N):
    dist_matrix[i][i] = 0
for i in range(N):
    for j in range(i + 1, N):  # i < j
        forbidden = all_keys_set - {key_nodes[i], key_nodes[j]}
        dist_ij = shortest_path(grid, key_nodes[i], key_nodes[j], forbidden)
        dist_matrix[i][j] = dist_ij
        dist_matrix[j][i] = dist_ij  # symmetric

# Now DP
INF = 10**9
dp = [[INF for _ in range(N)] for _ in range(1 << N)]
mask_s = 1 << S_idx
dp[mask_s][S_idx] = 0

for mask in range(0, 1 << N):
    for u in range(N):
        if (mask & (1 << u)) != 0:  # u in mask
            if mask == (1 << u) and u == S_idx:
                dp[mask][u] = 0  # base case
                continue
            prev_mask = mask ^ (1 << u)
            dp[mask][u] = INF  # set to inf before min
            for v in range(N):
                if (prev_mask & (1 << v)) != 0:  # v in prev_mask
                    if dp[prev_mask][v] < INF and dist_matrix[v][u] < INF:
                        dp[mask][u] = min(dp[mask][u], dp[prev_mask][v] + dist_matrix[v][u])

# Now find max num candies with dist to G <= T
ans = -1
for mask in range(1 << N):
    if (mask & (1 << G_idx)) != 0:  # G in mask
        dist_to_g = dp[mask][G_idx]
        if dist_to_g <= T:
            # compute num candies
            num_candy = 0
            temp_mask = mask
            while temp_mask:
                num_candy += temp_mask & 1
                temp_mask >>= 1
            num_candy -= 2  # subtract S and G bits
            if num_candy > ans:
                ans = num_candy

print(ans)