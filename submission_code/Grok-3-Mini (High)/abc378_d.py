import sys

# Read input
data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
K = int(data[index])
index += 1

# Read grid
grid = []
for _ in range(H):
    row = data[index]
    index += 1
    grid.append(row)

# Create visited matrix
visited = [[False for _ in range(W)] for _ in range(H)]

# Define DFS function
def dfs(i, j, rem):
    if rem == 0:
        return 1
    count = 0
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni = i + di
        nj = j + dj
        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.' and not visited[ni][nj]:
            visited[ni][nj] = True
            count += dfs(ni, nj, rem - 1)
            visited[ni][nj] = False
    return count

# Compute the answer
ans = 0
for start_i in range(H):
    for start_j in range(W):
        if grid[start_i][start_j] == '.':
            # Reset visited matrix
            for x in range(H):
                for y in range(W):
                    visited[x][y] = False
            # Mark start cell as visited
            visited[start_i][start_j] = True
            # Add the number of ways starting from this cell
            ans += dfs(start_i, start_j, K)

# Output the answer
print(ans)