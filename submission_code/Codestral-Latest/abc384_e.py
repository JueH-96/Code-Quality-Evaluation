import heapq

# Read input
H, W, X = map(int, input().split())
P, Q = map(int, input().split())
grid = [list(map(int, input().split())) for _ in range(H)]

# Directions for moving in the grid
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Priority queue to store slimes that can be absorbed
pq = []

# Initial strength of Takahashi
current_strength = grid[P-1][Q-1]

# Mark the initial position as visited
visited = [[False] * W for _ in range(H)]
visited[P-1][Q-1] = True

# Add initial position to the priority queue
heapq.heappush(pq, (-current_strength, P-1, Q-1))

while pq:
    strength, x, y = heapq.heappop(pq)
    strength = -strength

    # Check all four possible directions
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny]:
            if grid[nx][ny] < strength / X:
                # Absorb the slime
                current_strength += grid[nx][ny]
                visited[nx][ny] = True
                heapq.heappush(pq, (-current_strength, nx, ny))
                break

# Print the maximum possible strength of Takahashi
print(current_strength)