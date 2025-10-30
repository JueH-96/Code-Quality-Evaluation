import sys
from collections import deque

def main():
    input = sys.stdin.readline
    H, W, D = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]
    
    # Distance array, -1 means unvisited
    dist = [[-1]*W for _ in range(H)]
    dq = deque()
    
    # Initialize queue with all humidifiers
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                dist[i][j] = 0
                dq.append((i, j))
    
    # Directions: up, down, left, right
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    
    # Multi-source BFS
    while dq:
        x, y = dq.popleft()
        d0 = dist[x][y]
        if d0 == D:
            continue  # don't go beyond D
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W:
                if grid[nx][ny] != '#' and dist[nx][ny] == -1:
                    dist[nx][ny] = d0 + 1
                    dq.append((nx, ny))
    
    # Count humidified cells (excluding walls)
    ans = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '#' and dist[i][j] != -1:
                ans += 1
    
    print(ans)

if __name__ == "__main__":
    main()