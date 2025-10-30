from collections import deque
import sys
input = sys.stdin.read

def solve():
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    grid = data[2:H+2]
    A = int(data[H+2]) - 1
    B = int(data[H+3]) - 1
    C = int(data[H+4]) - 1
    D = int(data[H+5]) - 1
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    kick_range = [(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3) if abs(dx) + abs(dy) <= 2]
    
    # BFS with state (x, y, kicks)
    queue = deque([(A, B, 0)])
    visited = set([(A, B, 0)])
    
    while queue:
        x, y, kicks = queue.popleft()
        
        # If we reached the destination
        if (x, y) == (C, D):
            print(kicks)
            return
        
        # Move to adjacent cells
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '.':
                if (nx, ny, kicks) not in visited:
                    visited.add((nx, ny, kicks))
                    queue.append((nx, ny, kicks))
        
        # Try kicking in each direction
        for dx, dy in directions:
            new_kicks = kicks + 1
            modified = False
            temp_grid = [list(row) for row in grid]
            
            for kx, ky in kick_range:
                nx, ny = x + dx * kx, y + dy * ky
                if 0 <= nx < H and 0 <= ny < W and temp_grid[nx][ny] == '#':
                    temp_grid[nx][ny] = '.'
                    modified = True
            
            if modified:
                for kx, ky in kick_range:
                    nx, ny = x + dx * kx, y + dy * ky
                    if 0 <= nx < H and 0 <= ny < W and temp_grid[nx][ny] == '.':
                        if (nx, ny, new_kicks) not in visited:
                            visited.add((nx, ny, new_kicks))
                            queue.append((nx, ny, new_kicks))

solve()