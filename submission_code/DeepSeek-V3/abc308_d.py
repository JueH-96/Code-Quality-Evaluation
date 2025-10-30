from collections import deque

# Read input
H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Define the snuke sequence
snuke = ['s', 'n', 'u', 'k', 'e']

# Initialize BFS
visited = [[-1 for _ in range(W)] for _ in range(H)]
queue = deque()

# Check if the starting cell is 's'
if grid[0][0] != 's':
    print("No")
    exit()

# Initialize the queue with the starting position and step 0
queue.append((0, 0, 0))
visited[0][0] = 0

# Directions: up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Perform BFS
while queue:
    i, j, step = queue.popleft()
    current_char = grid[i][j]
    expected_char = snuke[step % 5]
    if current_char != expected_char:
        continue
    if i == H-1 and j == W-1:
        print("Yes")
        exit()
    for di, dj in directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < H and 0 <= nj < W:
            next_char = grid[ni][nj]
            next_step = (step + 1) % 5
            expected_next_char = snuke[next_step]
            if next_char == expected_next_char and visited[ni][nj] == -1:
                visited[ni][nj] = next_step
                queue.append((ni, nj, step + 1))

# If no path found
print("No")