from typing import List

class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        """
        Calculates the length of the longest V-shaped diagonal segment in a grid.
        
        This problem is solved using dynamic programming by breaking it down into three main parts:
        1.  Calculate lengths of all straight diagonal segments that start with 1. These form the first arm of a V-shape.
        2.  Calculate lengths of all straight diagonal "continuation" segments, which start with 0 or 2. These form the second arm.
        3.  Iterate through all cells as potential turning points, combining a "first arm" with a "second arm" to form a V-shape and find the maximum possible length.

        To implement this, we use 12 DP tables:
        - 4 tables for segments starting with 1 and ending at (r, c).
        - 8 tables for continuation segments (4 starting with 0, 4 starting with 2) that start at (r, c).

        The algorithm proceeds by:
        - First, calculating all straight segments starting with 1, which gives the answer for segments with zero turns.
        - Second, calculating all possible continuation segments.
        - Third, iterating through each cell, considering it a turn point, and combining the pre-calculated arm lengths to find the longest V-shape.
        """
        n = len(grid)
        m = len(grid[0])

        # dp1_end_DIR[r][c]: length of a valid segment starting with 1,
        # moving towards DIR, and ending at (r, c).
        dp1_end_tl_br = [[0] * m for _ in range(n)]
        dp1_end_tr_bl = [[0] * m for _ in range(n)]
        dp1_end_bl_tr = [[0] * m for _ in range(n)]
        dp1_end_br_tl = [[0] * m for _ in range(n)]

        # dp0_start_DIR[r][c]: length of a valid segment starting with 0,
        # starting at (r, c) and moving towards DIR.
        # dp2_start_DIR[r][c]: length of a valid segment starting with 2,
        # starting at (r, c) and moving towards DIR.
        dp0_start_tl_br = [[0] * m for _ in range(n)]
        dp2_start_tl_br = [[0] * m for _ in range(n)]
        dp0_start_tr_bl = [[0] * m for _ in range(n)]
        dp2_start_tr_bl = [[0] * m for _ in range(n)]
        dp0_start_bl_tr = [[0] * m for _ in range(n)]
        dp2_start_bl_tr = [[0] * m for _ in range(n)]
        dp0_start_br_tl = [[0] * m for _ in range(n)]
        dp2_start_br_tl = [[0] * m for _ in range(n)]
        
        max_len = 0

        # Phase 1: Calculate dp1_end (segments starting with 1)
        # from Top-Left (moves towards Bottom-Right)
        for r in range(n):
            for c in range(m):
                if grid[r][c] == 1:
                    dp1_end_tl_br[r][c] = 1
                elif r > 0 and c > 0:
                    prev_len = dp1_end_tl_br[r-1][c-1]
                    if prev_len > 0 and ((prev_len % 2 == 1 and grid[r][c] == 2) or (prev_len % 2 == 0 and grid[r][c] == 0)):
                        dp1_end_tl_br[r][c] = prev_len + 1
                if dp1_end_tl_br[r][c] > 0: max_len = max(max_len, dp1_end_tl_br[r][c])

        # from Top-Right (moves towards Bottom-Left)
        for r in range(n):
            for c in range(m - 1, -1, -1):
                if grid[r][c] == 1:
                    dp1_end_tr_bl[r][c] = 1
                elif r > 0 and c < m - 1:
                    prev_len = dp1_end_tr_bl[r-1][c+1]
                    if prev_len > 0 and ((prev_len % 2 == 1 and grid[r][c] == 2) or (prev_len % 2 == 0 and grid[r][c] == 0)):
                        dp1_end_tr_bl[r][c] = prev_len + 1
                if dp1_end_tr_bl[r][c] > 0: max_len = max(max_len, dp1_end_tr_bl[r][c])

        # from Bottom-Left (moves towards Top-Right)
        for r in range(n - 1, -1, -1):
            for c in range(m):
                if grid[r][c] == 1:
                    dp1_end_bl_tr[r][c] = 1
                elif r < n - 1 and c > 0:
                    prev_len = dp1_end_bl_tr[r+1][c-1]
                    if prev_len > 0 and ((prev_len % 2 == 1 and grid[r][c] == 2) or (prev_len % 2 == 0 and grid[r][c] == 0)):
                        dp1_end_bl_tr[r][c] = prev_len + 1
                if dp1_end_bl_tr[r][c] > 0: max_len = max(max_len, dp1_end_bl_tr[r][c])
        
        # from Bottom-Right (moves towards Top-Left)
        for r in range(n - 1, -1, -1):
            for c in range(m - 1, -1, -1):
                if grid[r][c] == 1:
                    dp1_end_br_tl[r][c] = 1
                elif r < n - 1 and c < m - 1:
                    prev_len = dp1_end_br_tl[r+1][c+1]
                    if prev_len > 0 and ((prev_len % 2 == 1 and grid[r][c] == 2) or (prev_len % 2 == 0 and grid[r][c] == 0)):
                        dp1_end_br_tl[r][c] = prev_len + 1
                if dp1_end_br_tl[r][c] > 0: max_len = max(max_len, dp1_end_br_tl[r][c])

        # Phase 2: Calculate dp0/2_start (continuation segments)
        # to Bottom-Right
        for r in range(n - 1, -1, -1):
            for c in range(m - 1, -1, -1):
                if grid[r][c] == 0: dp0_start_tl_br[r][c] = 1 + (dp2_start_tl_br[r+1][c+1] if r < n-1 and c < m-1 else 0)
                if grid[r][c] == 2: dp2_start_tl_br[r][c] = 1 + (dp0_start_tl_br[r+1][c+1] if r < n-1 and c < m-1 else 0)
        # to Bottom-Left
        for r in range(n - 1, -1, -1):
            for c in range(m):
                if grid[r][c] == 0: dp0_start_tr_bl[r][c] = 1 + (dp2_start_tr_bl[r+1][c-1] if r < n-1 and c > 0 else 0)
                if grid[r][c] == 2: dp2_start_tr_bl[r][c] = 1 + (dp0_start_tr_bl[r+1][c-1] if r < n-1 and c > 0 else 0)
        # to Top-Right
        for r in range(n):
            for c in range(m-1, -1, -1):
                if grid[r][c] == 0: dp0_start_bl_tr[r][c] = 1 + (dp2_start_bl_tr[r-1][c+1] if r > 0 and c < m-1 else 0)
                if grid[r][c] == 2: dp2_start_bl_tr[r][c] = 1 + (dp0_start_bl_tr[r-1][c+1] if r > 0 and c < m-1 else 0)
        # to Top-Left
        for r in range(n):
            for c in range(m):
                if grid[r][c] == 0: dp0_start_br_tl[r][c] = 1 + (dp2_start_br_tl[r-1][c-1] if r > 0 and c > 0 else 0)
                if grid[r][c] == 2: dp2_start_br_tl[r][c] = 1 + (dp0_start_br_tl[r-1][c-1] if r > 0 and c > 0 else 0)

        # Phase 3: Combine for V-shapes at each potential turn point (r,c)
        for r in range(n):
            for c in range(m):
                # Clockwise turns: (1,1)->(1,-1), (1,-1)->(-1,-1), (-1,-1)->(-1,1), (-1,1)->(1,1)
                
                # Turn 1: came from TL, moving (1,1), turns to move (1,-1)
                L1 = dp1_end_tl_br[r][c]
                if L1 > 0:
                    next_r, next_c = r + 1, c - 1
                    if 0 <= next_r < n and 0 <= next_c < m:
                        L2 = dp2_start_tr_bl[next_r][next_c] if L1 % 2 == 1 else dp0_start_tr_bl[next_r][next_c]
                        max_len = max(max_len, L1 + L2)
                
                # Turn 2: came from TR, moving (1,-1), turns to move (-1,-1)
                L1 = dp1_end_tr_bl[r][c]
                if L1 > 0:
                    next_r, next_c = r - 1, c - 1
                    if 0 <= next_r < n and 0 <= next_c < m:
                        L2 = dp2_start_br_tl[next_r][next_c] if L1 % 2 == 1 else dp0_start_br_tl[next_r][next_c]
                        max_len = max(max_len, L1 + L2)

                # Turn 3: came from BR, moving (-1,-1), turns to move (-1,1)
                L1 = dp1_end_br_tl[r][c]
                if L1 > 0:
                    next_r, next_c = r - 1, c + 1
                    if 0 <= next_r < n and 0 <= next_c < m:
                        L2 = dp2_start_bl_tr[next_r][next_c] if L1 % 2 == 1 else dp0_start_bl_tr[next_r][next_c]
                        max_len = max(max_len, L1 + L2)

                # Turn 4: came from BL, moving (-1,1), turns to move (1,1)
                L1 = dp1_end_bl_tr[r][c]
                if L1 > 0:
                    next_r, next_c = r + 1, c + 1
                    if 0 <= next_r < n and 0 <= next_c < m:
                        L2 = dp2_start_tl_br[next_r][next_c] if L1 % 2 == 1 else dp0_start_tl_br[next_r][next_c]
                        max_len = max(max_len, L1 + L2)
        
        return max_len