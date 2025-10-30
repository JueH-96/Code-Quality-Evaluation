class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = {}
        mod = 10**9 + 7

        def get_dp_value(r, c, current_xor):
            if (r, c, current_xor) in dp:
                return dp[(r, c, current_xor)]
            return 0

        dp[(0, 0, grid[0][0])] = 1

        for r in range(m):
            for c in range(n):
                for current_xor in range(16):
                    count = get_dp_value(r, c, current_xor)
                    if count > 0:
                        # Move right
                        if c + 1 < n:
                            next_xor = current_xor ^ grid[r][c + 1]
                            dp[(r, c + 1, next_xor)] = get_dp_value(r, c + 1, next_xor) + count
                            dp[(r, c + 1, next_xor)] %= mod
                        # Move down
                        if r + 1 < m:
                            next_xor = current_xor ^ grid[r + 1][c]
                            dp[(r + 1, c, next_xor)] = get_dp_value(r + 1, c, next_xor) + count
                            dp[(r + 1, c, next_xor)] %= mod

        return get_dp_value(m - 1, n - 1, k)