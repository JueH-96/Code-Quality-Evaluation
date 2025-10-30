def main():
    import sys
    sys.setrecursionlimit(10**6)
    from collections import deque

    data = sys.stdin.read().splitlines()
    if not data:
        return

    first_line = data[0].split()
    H = int(first_line[0])
    W = int(first_line[1])
    grid = data[1:1+H]
    
    # Create a 2D visited list to mark cells that have been explored.
    visited = [[False] * W for _ in range(H)]
    
    # These directions represent the 8 neighboring cells: horizontal, vertical, and diagonal.
    directions = [(-1, -1), (-1, 0), (-1, 1), 
                  (0, -1),           (0, 1), 
                  (1, -1),  (1, 0),  (1, 1)]
    
    sensor_groups = 0
    
    # We iterate through each cell on the grid.
    for i in range(H):
        for j in range(W):
            # Proceed only if there's a sensor at (i, j) and it hasn't been visited.
            if grid[i][j] == '#' and not visited[i][j]:
                sensor_groups += 1  # Found a new connected sensor group.
                # Use a stack for Depth-First Search (DFS) standard iterative implementation.
                stack = [(i, j)]
                visited[i][j] = True
                while stack:
                    x, y = stack.pop()
                    # Check all adjacent cells.
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and not visited[nx][ny]:
                                visited[nx][ny] = True
                                stack.append((nx, ny))
    
    # Print the number of distinct sensor groups.
    sys.stdout.write(str(sensor_groups))

if __name__ == '__main__':
    main()