# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    D = int(data[2])
    
    grid = data[3:H+3]
    
    from collections import deque
    
    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Visited array to mark humidified cells
    visited = [[False] * W for _ in range(H)]
    
    # BFS queue
    queue = deque()
    
    # Initialize the queue with all humidifier positions
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                queue.append((i, j, 0))  # (row, col, distance)
                visited[i][j] = True
    
    # Perform BFS
    while queue:
        x, y, dist = queue.popleft()
        
        # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny] and grid[nx][ny] != '#':
                if dist + 1 <= D:
                    visited[nx][ny] = True
                    queue.append((nx, ny, dist + 1))
    
    # Count humidified cells
    humidified_count = sum(visited[i][j] for i in range(H) for j in range(W))
    
    print(humidified_count)

if __name__ == "__main__":
    main()