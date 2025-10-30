def min_stairs_to_move(H, W, floors, queries):
    from collections import deque
    
    # Directions for adjacent blocks (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    results = []
    
    for A, B, Y, C, D, Z in queries:
        A -= 1  # Convert to 0-indexed
        B -= 1  # Convert to 0-indexed
        C -= 1  # Convert to 0-indexed
        D -= 1  # Convert to 0-indexed
        
        # BFS to find the minimum stairs used
        queue = deque()
        visited = set()
        
        # Start from the initial position
        queue.append((A, B, Y, 0))  # (row, col, floor, stairs_used)
        visited.add((A, B, Y))
        
        while queue:
            i, j, x, stairs_used = queue.popleft()
            
            # If we reach the destination
            if i == C and j == D:
                if x == Z:
                    results.append(stairs_used)
                    break
            
            # Move within the same building (up and down)
            if x < floors[i][j]:
                if (i, j, x + 1) not in visited:
                    visited.add((i, j, x + 1))
                    queue.append((i, j, x + 1, stairs_used + 1))
            if x > 1:
                if (i, j, x - 1) not in visited:
                    visited.add((i, j, x - 1))
                    queue.append((i, j, x - 1, stairs_used + 1))
            
            # Move to adjacent buildings
            for di, dj in directions:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    if floors[ni][nj] >= x:  # Can move to the x-th floor
                        if (ni, nj, x) not in visited:
                            visited.add((ni, nj, x))
                            queue.append((ni, nj, x, stairs_used))
        
    return results

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

H, W = map(int, data[0].split())
floors = [list(map(int, data[i + 1].split())) for i in range(H)]
Q = int(data[H + 1])
queries = [tuple(map(int, data[H + 2 + i].split())) for i in range(Q)]

# Get results
results = min_stairs_to_move(H, W, floors, queries)

# Print results
for result in results:
    print(result)