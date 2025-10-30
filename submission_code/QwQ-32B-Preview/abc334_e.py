import sys
from collections import deque

MOD = 998244353

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    
    # Directions for neighbors
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    # Perform BFS to find green connected components
    component_id = [[0]*W for _ in range(H)]
    component = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and component_id[i][j] == 0:
                component += 1
                q = deque()
                q.append((i, j))
                component_id[i][j] = component
                while q:
                    x, y = q.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and component_id[nx][ny] == 0:
                                component_id[nx][ny] = component
                                q.append((nx, ny))
    
    C_initial = component
    sum_m = 0
    K = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                neighbors = set()
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] == '#':
                            neighbors.add(component_id[ni][nj])
                m = len(neighbors)
                sum_m += m
                K += 1
    
    P = (C_initial + 1) * K - sum_m
    Q = K
    R = (P * pow(Q, MOD-2, MOD)) % MOD
    print(R)

if __name__ == '__main__':
    main()