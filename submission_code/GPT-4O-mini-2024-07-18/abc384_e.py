def max_strength_after_absorption(H, W, X, P, Q, strengths):
    from collections import deque
    
    # Convert P and Q to 0-indexed
    P -= 1
    Q -= 1
    
    # Directions for adjacent cells (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Initial strength of Takahashi
    current_strength = strengths[P][Q]
    
    # Queue for BFS
    queue = deque()
    queue.append((P, Q))
    
    # Visited set to keep track of absorbed slimes
    visited = set()
    visited.add((P, Q))
    
    # While there are cells to process
    while queue:
        x, y = queue.popleft()
        
        # Check adjacent cells
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if the new position is within bounds
            if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in visited:
                # Check if the slime can be absorbed
                if strengths[nx][ny] < current_strength / X:
                    # Absorb the slime
                    current_strength += strengths[nx][ny]
                    visited.add((nx, ny))
                    # Add the new position to the queue
                    queue.append((nx, ny))
    
    return current_strength

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

# Parse input
H, W, X = map(int, data[0].split())
P, Q = map(int, data[1].split())
strengths = [list(map(int, line.split())) for line in data[2:H+2]]

# Calculate the maximum strength
result = max_strength_after_absorption(H, W, X, P, Q, strengths)

# Print the result
print(result)