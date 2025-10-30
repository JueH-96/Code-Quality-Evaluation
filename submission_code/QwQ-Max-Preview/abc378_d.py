def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    start_cells = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                start_cells.append((i+1, j+1))  # Convert to 1-based indices

    def get_index(i, j):
        return (i-1) * W + (j-1)

    def dfs(i, j, mask, steps_remaining):
        if steps_remaining == 0:
            return 1
        total = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 1 <= ni <= H and 1 <= nj <= W:
                idx = get_index(ni, nj)
                if (mask & (1 << idx)) == 0 and grid[ni-1][nj-1] == '.':
                    total += dfs(ni, nj, mask | (1 << idx), steps_remaining - 1)
        return total

    total = 0
    for i, j in start_cells:
        initial_mask = 1 << get_index(i, j)
        total += dfs(i, j, initial_mask, K)
    print(total)

if __name__ == '__main__':
    main()