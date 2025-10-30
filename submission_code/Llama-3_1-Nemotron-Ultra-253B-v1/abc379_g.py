MOD = 998244353

H, W = map(int, input().split())
grid = [input().strip() for _ in range(H)]

# Precompute allowed values for each cell
allowed = []
for i in range(H):
    row = []
    for j in range(W):
        c = grid[i][j]
        if c == '?':
            row.append([1, 2, 3])
        else:
            row.append([int(c)])
    allowed.append(row)

from functools import lru_cache

@lru_cache(maxsize=None)
def dfs(i, j, prev_row, curr):
    if i == H:
        return 1
    if j == W:
        return dfs(i + 1, 0, tuple(curr), ())
    # Determine the previous value from the previous row
    prev_val = prev_row[j] if i > 0 else -1
    # Determine the left value from the current row
    left_val = curr[-1] if j > 0 else -1
    res = 0
    for val in allowed[i][j]:
        if val == prev_val or val == left_val:
            continue
        new_curr = list(curr)
        new_curr.append(val)
        res += dfs(i, j + 1, prev_row, tuple(new_curr))
        res %= MOD
    return res % MOD

# Initial call: process first row, no previous row, empty current row
print(dfs(0, 0, (), ()))