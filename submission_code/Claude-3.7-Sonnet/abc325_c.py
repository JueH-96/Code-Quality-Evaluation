from collections import deque

def count_sensors():
    # Read input
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]
    
    # Initialize visited array
    visited = [[False for _ in range(W)] for _ in range(H)]
    sensor_count = 0
    
    # For each cell in the grid
    for i in range(H):
        for j in range(W):
            # If it has a sensor and hasn't been visited yet
            if grid[i][j] == '#' and not visited[i][j]:
                # Found a new sensor group
                sensor_count += 1
                
                # BFS to mark all connected sensors
                queue = deque([(i, j)])
                visited[i][j] = True
                
                while queue:
                    x, y = queue.popleft()
                    
                    # Check all 8 adjacent squares (horizontal, vertical, diagonal)
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            
                            nx, ny = x + dx, y + dy
                            
                            # If in bounds, has a sensor, and not visited yet
                            if (0 <= nx < H and 0 <= ny < W and 
                                grid[nx][ny] == '#' and not visited[nx][ny]):
                                visited[nx][ny] = True
                                queue.append((nx, ny))
    
    return sensor_count

if __name__ == "__main__":
    print(count_sensors())