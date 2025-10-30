from typing import List

class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        n = len(grid)
        m = len(grid[0])
        dir_delta = [(1,1), (1,-1), (-1,-1), (-1,1)]  # 0: DR, 1: DL, 2: UL, 3: UR

        # Precompute len_values
        len_values = [[[[0 for _ in range(m)] for _ in range(n)] for _ in range(2)] for _ in range(4)]  # [dir_idx][parity][r][c]  parity 0: even, 1: odd

        for dir_idx in range(4):
            dr, dc = dir_delta[dir_idx]
            for r in range(n):
                for c in range(m):
                    for parity in range(2):
                        len_values[dir_idx][parity][r][c] = self.get_len(grid, r, c, dr, dc, parity, n, m)

        max_length = 0

        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    for dir1_idx in range(4):
                        dr1, dc1 = dir_delta[dir1_idx]
                        # Compute straight_len for no-turn
                        if 0 <= i + dr1 < n and 0 <= j + dc1 < m and grid[i + dr1][j + dc1] == 2:
                            straight_len = 1 + len_values[dir1_idx][1][i + dr1][j + dc1]  # k=1 odd
                        else:
                            straight_len = 1
                        max_length = max(max_length, straight_len)

                        # Now for turn case
                        if straight_len > 1:
                            for T in range(1, straight_len):  # T from 1 to straight_len-1
                                new_dir_idx = (dir1_idx + 1) % 4
                                new_dr = dc1
                                new_dc = -dr1
                                nr = i + T * dr1 + new_dr
                                nc = j + T * dc1 + new_dc
                                k_next = T + 1
                                L_second = 0
                                if 0 <= nr < n and 0 <= nc < m:
                                    if k_next % 2 == 1:
                                        if grid[nr][nc] == 2:
                                            L_second = len_values[new_dir_idx][1][nr][nc]
                                    else:
                                        if grid[nr][nc] == 0:
                                            L_second = len_values[new_dir_idx][0][nr][nc]
                                total_len = (T + 1) + L_second
                                max_length = max(max_length, total_len)

        return max_length

    def get_len(self, grid, r, c, dr, dc, parity, n, m):
        if parity == 1:  # k odd
            if grid[r][c] != 2:
                return 0
        else:  # k even
            if grid[r][c] != 0:
                return 0
        len_val = 1
        nr, nc = r + dr, c + dc
        if not (0 <= nr < n and 0 <= nc < m):
            return 1
        expected_val = 0 if parity == 1 else 2  # if parity odd, next k even, value 0; if even, next k odd, value 2
        if grid[nr][nc] == expected_val:
            len_val += 1
            nr2 = nr + dr
            nc2 = nc + dc
            expected_next = 2 if expected_val == 0 else 0  # flip
            while 0 <= nr2 < n and 0 <= nc2 < m and grid[nr2][nc2] == expected_next:
                len_val += 1
                nr2 += dr
                nc2 += dc
                expected_next = 2 if expected_next == 0 else 0
            return len_val
        else:
            return 1  # only the start cell