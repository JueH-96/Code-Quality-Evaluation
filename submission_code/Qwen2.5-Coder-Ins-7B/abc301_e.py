import sys
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    T = int(data[2])
    grid = data[3:H+3]
    
    start = None
    goal = None
    candies = []
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'G':
                goal = (i, j)
            elif grid[i][j] == 'o':
                candies.append((i, j))
    
    if start is None or goal is None:
        print(-1)
        return
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = [[[False] * (T + 1) for _ in range(W)] for _ in range(H)]
    queue = deque([(start[0], start[1], 0, 0)])
    visited[start[0]][start[1]][0] = True
    
    while queue:
        x, y, t, candies_collected = queue.popleft()
        
        if (x, y) == goal:
            print(candies_collected)
            return
        
        if t >= T:
            continue
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < H and 0 <= ny < W and not visited[nx][ny][t + 1]:
                if grid[nx][ny] != '#':
                    visited[nx][ny][t + 1] = True
                    if grid[nx][ny] == 'o':
                        queue.append((nx, ny, t + 1, candies_collected + 1))
                    else:
                        queue.append((nx, ny, t + 1, candies_collected))
    
    print(-1)

if __name__ == "__main__":
    main()