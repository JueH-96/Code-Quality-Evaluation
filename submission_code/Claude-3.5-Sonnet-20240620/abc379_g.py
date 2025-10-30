# YOUR CODE HERE
MOD = 998244353

def count_valid_grids(H, W, grid):
    def is_valid(i, j, val):
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] != '?' and grid[ni][nj] == val:
                    return False
        return True

    def dfs(i, j):
        if i == H:
            return 1
        if j == W:
            return dfs(i + 1, 0)
        
        if grid[i][j] != '?':
            return dfs(i, j + 1)
        
        count = 0
        for val in '123':
            if is_valid(i, j, val):
                grid[i][j] = val
                count = (count + dfs(i, j + 1)) % MOD
                grid[i][j] = '?'
        return count

    return dfs(0, 0)

H, W = map(int, input().split())
grid = [list(input().strip()) for _ in range(H)]

result = count_valid_grids(H, W, grid)
print(result)