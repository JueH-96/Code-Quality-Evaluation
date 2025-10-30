from collections import deque

def min_front_kicks(H, W, grid, A, B, C, D):
    # Directions for moving and kicking
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Initialize the BFS queue
    queue = deque([(A, B, 0)])  # (x, y, number of kicks)
    visited = [[float('inf')] * W for _ in range(H)]
    visited[A][B] = 0
    
    while queue:
        x, y, kicks = queue.popleft()
        
        # If we reach the destination
        if (x, y) == (C, D):
            return kicks
        
        # Move to adjacent road cells
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '.':
                if visited[nx][ny] > kicks:
                    visited[nx][ny] = kicks
                    queue.appendleft((nx, ny, kicks))  # Prioritize moving without kicking
        
        # Perform front kicks in all four directions
        for dx, dy in directions:
            for step in range(1, 3):  # Kick affects up to 2 steps away
                nx, ny = x + dx * step, y + dy * step
                if 0 <= nx < H and 0 <= ny < W:
                    if grid[nx][ny] == '#' and visited[nx][ny] > kicks + 1:
                        visited[nx][ny] = kicks + 1
                        queue.append((nx, ny, kicks + 1))
                    elif grid[nx][ny] == '.' and visited[nx][ny] > kicks:
                        visited[nx][ny] = kicks
                        queue.appendleft((nx, ny, kicks))
    
    return -1  # If unreachable, though problem guarantees it's reachable

# Read input
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
grid = data[2:H+2]
A = int(data[H+2]) - 1
B = int(data[H+3]) - 1
C = int(data[H+4]) - 1
D = int(data[H+5]) - 1

# Solve the problem
result = min_front_kicks(H, W, grid, A, B, C, D)
print(result)