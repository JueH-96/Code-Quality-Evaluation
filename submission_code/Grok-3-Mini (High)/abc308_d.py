import sys
import collections

# Read input
data = sys.stdin.read().split()
index = 0
H = int(data[index])
W = int(data[index + 1])
index += 2
grid = [data[index + i] for i in range(H)]

# Required letters for each mod value (t % 5)
req = ['e', 's', 'n', 'u', 'k']  # Index 0-4 correspond to t%5 = 0,1,2,3,4

# BFS setup
visited = [[[False for _ in range(5)] for _ in range(W)] for _ in range(H)]
queue = collections.deque()

# Start from (0,0) with mod 1 (t=1) if the cell matches 's'
if grid[0][0] == req[1]:  # req[1] is 's'
    visited[0][0][1] = True
    queue.append((0, 0, 1))

found = False

# BFS
while queue:
    ci, cj, cm = queue.popleft()
    # Check if reached the bottom-right cell
    if ci == H - 1 and cj == W - 1:
        found = True
        break
    # Possible moves: up, right, down, left
    for di, dj in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
        ni = ci + di
        nj = cj + dj
        # Check bounds
        if 0 <= ni < H and 0 <= nj < W:
            new_mod = (cm + 1) % 5  # New mod value after moving
            # Check if the cell has the correct letter and not visited
            if grid[ni][nj] == req[new_mod] and not visited[ni][nj][new_mod]:
                visited[ni][nj][new_mod] = True
                queue.append((ni, nj, new_mod))

# Output result
if found:
    print("Yes")
else:
    print("No")