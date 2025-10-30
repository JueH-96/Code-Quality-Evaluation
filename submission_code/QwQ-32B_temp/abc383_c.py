import sys
from collections import deque

def main():
    H, W, D = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        grid.append(sys.stdin.readline().strip())
    
    h_positions = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'H':
                h_positions.append((i, j))
    
    distance = [[-1 for _ in range(W)] for _ in range(H)]
    q = deque()
    
    for i, j in h_positions:
        distance[i][j] = 0
        q.append((i, j))
    
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while q:
        i, j = q.popleft()
        for dx, dy in dirs:
            ni = i + dx
            nj = j + dy
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] != '#' and distance[ni][nj] == -1:
                    new_dist = distance[i][j] + 1
                    distance[ni][nj] = new_dist
                    if new_dist <= D:
                        q.append((ni, nj))
    
    count = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                continue
            if distance[i][j] != -1 and distance[i][j] <= D:
                count += 1
    print(count)

if __name__ == "__main__":
    main()