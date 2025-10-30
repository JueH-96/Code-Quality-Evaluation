def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    
    H, W = map(int, input_data[:2])
    grid_lines = input_data[2:]
    
    # Grid of booleans: True if there's a sensor ('#'), else False
    grid = []
    for i in range(H):
        row = grid_lines[i]
        grid.append([c == '#' for c in row])
    
    visited = [[False]*W for _ in range(H)]
    
    # Directions (8-neighborhood: horizontal, vertical, diagonal)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]
    
    def bfs(start_i, start_j):
        from collections import deque
        queue = deque()
        queue.append((start_i, start_j))
        visited[start_i][start_j] = True
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    if grid[nx][ny] and not visited[nx][ny]:
                        visited[nx][ny] = True
                        queue.append((nx, ny))
    
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] and not visited[i][j]:
                # Found a new sensor cluster
                bfs(i, j)
                count += 1
    
    print(count)

# Do not forget to call main
if __name__ == "__main__":
    main()