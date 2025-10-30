# YOUR CODE HERE
from collections import deque

def solve():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]
    N = int(input())
    medicines = [tuple(map(int, input().split())) for _ in range(N)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    visited = [[[False for _ in range(W+1)] for _ in range(H+1)] for _ in range(W*H+1)]
    queue = deque([(start[0], start[1], 0)])
    visited[0][start[0]][start[1]] = True
    
    while queue:
        x, y, energy = queue.popleft()
        if (x, y) == goal:
            print('Yes')
            return
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                next_energy = energy + 1
                if not visited[next_energy][nx][ny]:
                    visited[next_energy][nx][ny] = True
                    queue.append((nx, ny, next_energy))
        
        for rx, cx, e in medicines:
            if (x, y) == (rx-1, cx-1):
                next_energy = e
                if not visited[next_energy][x][y]:
                    visited[next_energy][x][y] = True
                    queue.append((x, y, next_energy))
    
    print('No')

solve()