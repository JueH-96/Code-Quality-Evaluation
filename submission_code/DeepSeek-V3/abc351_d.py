from collections import deque

def main():
    H, W = map(int, input().split())
    grid = [list(input().strip()) for _ in range(H)]
    
    # Directions for adjacent cells
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # To keep track of visited cells
    visited = [[False for _ in range(W)] for _ in range(H)]
    
    max_freedom = 0
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.' and not visited[i][j]:
                # Initialize BFS
                queue = deque()
                queue.append((i, j))
                visited[i][j] = True
                freedom = 0
                
                while queue:
                    x, y = queue.popleft()
                    freedom += 1
                    
                    # Check if any adjacent cell has a magnet
                    has_magnet = False
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#':
                                has_magnet = True
                                break
                    
                    if not has_magnet:
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < H and 0 <= ny < W:
                                if grid[nx][ny] == '.' and not visited[nx][ny]:
                                    visited[nx][ny] = True
                                    queue.append((nx, ny))
                
                if freedom > max_freedom:
                    max_freedom = freedom
    
    print(max_freedom)

if __name__ == "__main__":
    main()