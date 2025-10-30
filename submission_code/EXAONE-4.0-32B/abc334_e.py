mod = 998244353

import collections
import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    comp_id = [[-1] * W for _ in range(H)]
    total_green_components = 0
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    queue = collections.deque()
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and comp_id[i][j] == -1:
                total_green_components += 1
                comp_id[i][j] = total_green_components - 1
                queue.append((i, j))
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and comp_id[nx][ny] == -1:
                                comp_id[nx][ny] = total_green_components - 1
                                queue.append((nx, ny))
    
    n_red = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                n_red += 1
                
    S = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                seen = set()
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W:
                        if grid[ni][nj] == '#':
                            cid = comp_id[ni][nj]
                            seen.add(cid)
                k_val = len(seen)
                S += k_val
                
    numerator = (total_green_components * n_red + n_red - S) % mod
    inv_n_red = pow(n_red, mod-2, mod)
    ans = numerator * inv_n_red % mod
    print(ans)

if __name__ == '__main__':
    main()