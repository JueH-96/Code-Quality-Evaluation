import sys

def main():
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [list(sys.stdin.readline().strip()) for _ in range(H)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def dfs(i, j, steps, mask):
        if steps == K:
            return 1
        total = 0
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '.':
                    pos = ni * W + nj
                    if not (mask & (1 << pos)):
                        new_mask = mask | (1 << pos)
                        total += dfs(ni, nj, steps + 1, new_mask)
        return total
    
    ans = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                pos = i * W + j
                ans += dfs(i, j, 0, 1 << pos)
    print(ans)

if __name__ == "__main__":
    main()