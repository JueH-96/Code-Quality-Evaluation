import sys
import collections

# Read input
data = sys.stdin.read().split()
H = int(data[0])
W = int(data[1])
grid = data[2:2 + H]

# Create component ID grid, initialized to -1
comp_id = [[-1 for _ in range(W)] for _ in range(H)]

# Component count, which is C_old
comp_count = 0

# BFS function to assign component ID
def bfs(grid, comp_id, start_i, start_j, cid, H, W):
    queue = collections.deque()
    queue.append((start_i, start_j))
    comp_id[start_i][start_j] = cid
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#' and comp_id[nx][ny] == -1:
                comp_id[nx][ny] = cid
                queue.append((nx, ny))

# Assign components to all green cells
for i in range(H):
    for j in range(W):
        if grid[i][j] == '#' and comp_id[i][j] == -1:
            bfs(grid, comp_id, i, j, comp_count, H, W)
            comp_count += 1

# Now compute R and sum of M_i
R = 0  # Number of red cells
sum_M = 0  # Sum of M_i over all red cells
for i in range(H):
    for j in range(W):
        if grid[i][j] == '.':  # Red cell
            R += 1
            adj_comps = set()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx = i + dx
                ny = j + dy
                if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#':
                    comp = comp_id[nx][ny]
                    adj_comps.add(comp)
            M_i = len(adj_comps)
            sum_M += M_i

# Compute N and D
N = (comp_count + 1) * R - sum_M
D = R

# Modulo value
MOD = 998244353

# Compute N mod MOD, handle negative values
N_mod = (N % MOD + MOD) % MOD

# Compute modular inverse of D using Fermat's Little Theorem (since MOD is prime)
inv_D = pow(D, MOD - 2, MOD)

# Compute the result
result = (N_mod * inv_D) % MOD

# Print the result
print(result)