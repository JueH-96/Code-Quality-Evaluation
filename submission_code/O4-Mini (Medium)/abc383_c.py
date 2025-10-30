import sys
from collections import deque

def main():
    input = sys.stdin.readline
    H, W, D = map(int, input().split())
    grid = [input().rstrip('
') for _ in range(H)]
    
    # visited array to mark humidified cells
    visited = [[False] * W for _ in range(H)]
    q = deque()
    
    # Multi-source BFS initialization with all humidifiers
    for i in range(H):
        row = grid[i]
        for j in range(W):
            if row[j] == 'H':
                visited[i][j] = True
                q.append((i, j, 0))
    
    # Directions: up, down, left, right
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    
    # BFS up to distance D
    while q:
        i, j, d = q.popleft()
        if d == D:
            continue
        nd = d + 1
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if not visited[ni][nj] and grid[ni][nj] != '#':
                    visited[ni][nj] = True
                    q.append((ni, nj, nd))
    
    # Count humidified floor cells (including cells with 'H')
    ans = 0
    for i in range(H):
        for j in range(W):
            if visited[i][j]:
                # it's either '.' or 'H' since we never visit walls
                ans += 1
    
    print(ans)

if __name__ == "__main__":
    main()