import sys
from collections import deque

MOD = 998244353

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = []
    for _ in range(H):
        line = sys.stdin.readline().strip()
        grid.append(list(line))
    
    component = [[-1 for _ in range(W)] for _ in range(H)]
    current_component = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and component[i][j] == -1:
                queue = deque()
                queue.append((i, j))
                component[i][j] = current_component
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and component[nx][ny] == -1:
                                component[nx][ny] = current_component
                                queue.append((nx, ny))
                current_component += 1
    
    C_initial = current_component
    red_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                red_cells.append((i, j))
    
    sum_delta = 0
    for (i, j) in red_cells:
        components_set = set()
        for dx, dy in directions:
            ni = i + dx
            nj = j + dy
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '#':
                    components_set.add(component[ni][nj])
        k_i = len(components_set)
        sum_delta += (1 - k_i)
    
    K = len(red_cells)
    numerator = (C_initial * K + sum_delta) % MOD
    inv_K = pow(K, MOD - 2, MOD)
    result = (numerator * inv_K) % MOD
    print(result)

if __name__ == '__main__':
    main()