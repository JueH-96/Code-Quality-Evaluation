from collections import deque

mod = 998244353

def main():
    import sys
    data = sys.stdin.read().splitlines()
    H, W = map(int, data[0].split())
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    n_red = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                n_red += 1
                
    comp = [[-1] * W for _ in range(H)]
    comp_id = 0
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and comp[i][j] == -1:
                q = deque()
                q.append((i, j))
                comp[i][j] = comp_id
                while q:
                    x, y = q.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny]=='#' and comp[nx][ny] == -1:
                            comp[nx][ny] = comp_id
                            q.append((nx, ny))
                comp_id += 1
                
    C0 = comp_id
    total_k = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                seen_comps = set()
                for dx, dy in directions:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj]=='#':
                        seen_comps.add(comp[ni][nj])
                total_k += len(seen_comps)
                
    if n_red == 0:
        print(0)
    else:
        numerator = (n_red * (C0 + 1) - total_k) % mod
        inv_n = pow(n_red, mod-2, mod)
        ans = numerator * inv_n % mod
        print(ans)

if __name__ == "__main__":
    main()