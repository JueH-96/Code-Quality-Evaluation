import sys
sys.setrecursionlimit(1 << 25)

def main():
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    count = 0

    def dfs(i, j, mask, steps):
        if steps == 0:
            return 1
        res = 0
        # Directions: up, down, left, right
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni = i + dx
            nj = j + dy
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '.':
                    pos = ni * W + nj
                    if not (mask & (1 << pos)):
                        res += dfs(ni, nj, mask | (1 << pos), steps - 1)
        return res

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                pos = i * W + j
                mask = 1 << pos
                count += dfs(i, j, mask, K)
    print(count)

if __name__ == "__main__":
    main()