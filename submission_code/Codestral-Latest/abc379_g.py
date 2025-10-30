MOD = 998244353

def count_valid_grids(H, W, grid):
    from functools import lru_cache

    @lru_cache(None)
    def dp(i, j, prev):
        if i == H:
            return 1
        if j == W:
            return dp(i + 1, 0, -1)
        if grid[i][j] != '?':
            if grid[i][j] == prev:
                return 0
            return dp(i, j + 1, grid[i][j])
        total = 0
        for ch in '123':
            if ch != prev:
                total = (total + dp(i, j + 1, ch)) % MOD
        return total

    return dp(0, 0, -1)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    H = int(data[0])
    W = int(data[1])
    grid = [data[i] for i in range(2, H + 2)]

    result = count_valid_grids(H, W, grid)
    print(result)

if __name__ == "__main__":
    main()