class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        ans = 0

        def isValid(r, c):
            return 0 <= r < n and 0 <= c < m

        def get_len(r, c, dr, dc, turn, expected):
            length = 0
            while isValid(r, c) and grid[r][c] == expected:
                length += 1
                r += dr
                c += dc
                expected = (expected + 1) % 3
            return length, r, c, dr, dc, expected

        def solve(r, c, dr, dc):
            nonlocal ans
            if grid[r][c] != 1:
                return
            
            len1, r1, c1, dr1, dc1, expected1 = get_len(r, c, dr, dc, 0, 1)
            if len1 == 0:
                return

            ans = max(ans, len1)

            # Try turning
            directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dr2, dc2 in directions:
                if (dr2, dc2) != (dr, dc) and (dr2, dc2) != (-dr, -dc):
                    len2, _, _, _, _, _ = get_len(r1 - dr1, c1 - dc1, dr2, dc2, 1, expected1)
                    ans = max(ans, len1 + len2)

        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    solve(i, j, 1, 1)
                    solve(i, j, 1, -1)
                    solve(i, j, -1, 1)
                    solve(i, j, -1, -1)
        return ans