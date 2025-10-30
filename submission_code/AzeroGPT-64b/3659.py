from collections import defaultdict


class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        mod = 1000000007
        paths = defaultdict(int)
        paths[(0, 0, grid[0][0] if grid[0][0] < 16 else grid[0][0] % 16)] = 1

        for r in range(m):
            for c in range(n):
                if r == 0 and c == 0:
                    continue
                new_paths = defaultdict(int)
                value = grid[r][c] if grid[r][c] < 16 else grid[r][c] % 16
                for prev_r, prev_c, xor in paths.keys():
                    new_val = xor ^ value
                    if r > prev_r and (prev_c == c or prev_c + 1 == c):
                        new_paths[(r, c, new_val)] += paths[(prev_r, prev_c, xor)]
                    elif c > prev_c and (prev_r == r or prev_r + 1 == r):
                        new_paths[(r, c, new_val)] += paths[(prev_r, prev_c, xor)]
                paths = new_paths
        return paths[(m - 1, n - 1, k)] % mod