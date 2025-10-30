import sys
from collections import deque

def main():
    H, W, D = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    q = deque()
    distance = [[-1] * W for _ in range(H)]
    
    # Initialize queue with all H positions and set their distance to 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                distance[i][j] = 0
                q.append((i, j))
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Perform BFS
    while q:
        x, y = q.popleft()
        for dx, dy in directions:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < H and 0 <= ny < W:
                if grid[nx][ny] != '#' and distance[nx][ny] == -1:
                    distance[nx][ny] = distance[x][y] + 1
                    q.append((nx, ny))
    
    # Count the number of cells that are floors and have distance <= D
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '#' and distance[i][j] != -1 and distance[i][j] <= D:
                count += 1
    print(count)

if __name__ == "__main__":
    main()