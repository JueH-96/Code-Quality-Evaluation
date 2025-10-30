from collections import deque

# Read input
H, W, T = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Find start and goal positions
start = goal = None
candies = []
for i in range(H):
    for j in range(W):
        if grid[i][j] == 'S':
            start = (i, j)
        elif grid[i][j] == 'G':
            goal = (i, j)
        elif grid[i][j] == 'o':
            candies.append((i, j))

# Directions for moving up, down, left, right
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# BFS to find the shortest path to the goal
def bfs(start, goal):
    queue = deque([(start[0], start[1], 0, set())])
    visited = set()
    visited.add((start[0], start[1]))

    while queue:
        x, y, moves, visited_candies = queue.popleft()

        if (x, y) == goal:
            return moves, visited_candies

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and (nx, ny) not in visited:
                new_visited_candies = visited_candies.copy()
                if grid[nx][ny] == 'o':
                    new_visited_candies.add((nx, ny))
                visited.add((nx, ny))
                queue.append((nx, ny, moves + 1, new_visited_candies))

    return float('inf'), set()

# Find the shortest path to the goal
shortest_path_length, visited_candies = bfs(start, goal)

# Check if the goal can be reached within T moves
if shortest_path_length <= T:
    print(len(visited_candies))
else:
    print(-1)