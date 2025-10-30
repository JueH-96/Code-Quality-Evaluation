from collections import deque

H, W = map(int, input().split())
grid = [list(input().strip()) for _ in range(H)]
N = int(input())
medicines = []
for _ in range(N):
    R, C, E = map(int, input().split())
    medicines.append((R-1, C-1, E))  # converting to 0-based index

# Find start and goal positions
start = None
goal = None
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'T':
            goal = (i, j)
if not start or not goal:
    print("No")
    exit()

# Initialize visited and energy
visited = [[-1 for _ in range(W)] for _ in range(H)]
queue = deque()
queue.append((start[0], start[1], 0))  # (i, j, energy)
visited[start[0]][start[1]] = 0

# Preprocess medicines: map (i,j) to E
med_map = {}
for med in medicines:
    i, j, e = med
    if (i, j) not in med_map or med_map[(i, j)] < e:
        med_map[(i, j)] = e

# Directions for movement
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

while queue:
    i, j, e = queue.popleft()
    if (i, j) == goal:
        print("Yes")
        exit()
    # Check if current cell has medicine
    if (i, j) in med_map:
        new_e = med_map[(i, j)]
        if new_e > e:
            e = new_e
            visited[i][j] = e
    # Move to adjacent cells
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '#':
            if e > 0 and (visited[ni][nj] < e - 1):
                visited[ni][nj] = e - 1
                queue.append((ni, nj, e - 1))

print("No")