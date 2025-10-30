from typing import List

class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])

        # dp_straight[r][c][d_idx] = length of the longest straight segment ending at (r, c),
        # whose last step was arriving from d_idx. Valid for length >= 2.
        dp_straight = [[[0] * 4 for _ in range(m)] for _ in range(n)]

        # dp_v_end[r][c][d_idx] = length of the longest V-shaped segment ending at (r, c),
        # whose last step was arriving from d_idx, where the turn happened *before* reaching (r,c).
        # Valid for length >= 3.
        dp_v_end = [[[0] * 4 for _ in range(m)] for _ in range(n)]

        max_length = 0

        # Directions (dr, dc) from previous cell to current cell
        # 0: DR (+1, +1), 1: DL (+1, -1), 2: UL (-1, -1), 3: UR (-1, +1)
        dirs = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
        # Reverse clockwise turn mapping: rcw[current_d_idx] = prev_d_idx before turn
        # If current step arrived from DR (0), and turned at the previous cell, it must have arrived at the previous cell from UR (3).
        # If current step arrived from DL (1), and turned at the previous cell, it must have arrived at the previous cell from DR (0).
        # If current step arrived from UL (2), and turned at the previous cell, it must have arrived at the previous cell from DL (1).
        # If current step arrived from UR (3), and turned at the previous cell, it must have arrived at the previous cell from UL (2).
        # This corresponds to a clockwise turn of the *direction vector*.
        # DR(1,1) -> DL(1,-1) -> UL(-1,-1) -> UR(-1,1) -> DR(1,1)
        # Index 0 -> 1 -> 2 -> 3 -> 0.
        # So if current direction is d_idx, the previous direction (before the turn) was rcw[d_idx].
        rcw = [3, 0, 1, 2]

        # Function to get required value for overall segment length k (1-indexed, k > 1)
        # Segment sequence: 1 (step 1), 2 (step 2), 0 (step 3), 2 (step 4), 0 (step 5), ...
        # For k > 1: required value is 2 if k is even, 0 if k is odd.
        def get_required_value(k):
            # This pattern `2 if k % 2 == 0 else 0` works for k >= 2.
            return 2 if k % 2 == 0 else 0

        # Base case: Any cell with value 1 is a segment of length 1.
        for r in range(n):
            for c in range(m):
                if grid[r][c] == 1:
                    max_length = max(max_length, 1)

        # Iterate through possible ARRIVING directions d_idx at the current cell (r, c).
        for d_idx in range(4):
            dr, dc = dirs[d_idx]
            # Iterate through the grid in an order suitable for the dependency on the previous cell (r-dr, c-dc)
            # E.g., for DR (1,1), iterate r 0..n-1, c 0..m-1
            # E.g., for DL (1,-1), iterate r 0..n-1, c m-1..0
            # E.g., for UL (-1,-1), iterate r n-1..0, c m-1..0
            # E.g., for UR (-1,1), iterate r n-1..0, c 0..m-1
            r_range = range(n) if dr == 1 else range(n - 1, -1, -1)
            c_range = range(m) if dc == 1 else range(m - 1, -1, -1)

            for r in r_range:
                for c in c_range:
                    # Current cell is (r, c). Previous cell in this direction is (pr, pc).
                    pr, pc = r - dr, c - dc

                    # --- Calculate dp_straight[r][c][d_idx] (Length >= 2, arrived from d_idx, no turn) ---
                    # Requires (pr, pc) to be valid.
                    if 0 <= pr < n and 0 <= pc < m:
                        # Option A: Extend a straight segment of length >= 2 ending at (pr, pc), arrived from d_idx.
                        if dp_straight[pr][pc][d_idx] > 0:
                            L_prev = dp_straight[pr][pc][d_idx] # Length ending at (pr, pc), >= 2
                            # Current cell (r, c) is step L_prev + 1 (1-indexed overall step number)
                            if grid[r][c] == get_required_value(L_prev + 1):
                                dp_straight[r][c][d_idx] = L_prev + 1
                                max_length = max(max_length, dp_straight[r][c][d_idx])
                        # Option B: Extend a straight segment of length 1 starting at (pr, pc) (where grid[pr][pc] == 1).
                        elif grid[pr][pc] == 1:
                             # Current cell (r, c) is the second cell (overall length 2). Required value is 2.
                             if grid[r][c] == 2:
                                 dp_straight[r][c][d_idx] = 2
                                 max_length = max(max_length, dp_straight[r][c][d_idx])


                    # --- Calculate dp_v_end[r][c][d_idx] (Length >= 3, arrived from d_idx, turn happened before (r,c)) ---

                    # Option A: Extend a V-shaped segment of length >= 3 ending at (pr, pc),
                    #           arrived from d_idx (turn before pr,pc).
                    if 0 <= pr < n and 0 <= pc < m and dp_v_end[pr][pc][d_idx] > 0:
                        L_prev = dp_v_end[pr][pc][d_idx] # Length ending at (pr, pc), >= 3
                        # Current cell (r, c) is step L_prev + 1
                        if grid[r][c] == get_required_value(L_prev + 1):
                            dp_v_end[r][c][d_idx] = L_prev + 1
                            max_length = max(max_length, dp_v_end[r][c][d_idx])

                    # Option B: The segment turned AT (pr, pc).
                    # It arrived at (pr, pc) from some `(pr_turn, pc_turn)` via `prev_d_idx` (no turn),
                    # and then turned into direction `d_idx` to reach (r, c).
                    # The step from (pr, pc) to (r, c) is in direction `d_idx`.
                    # The previous direction `prev_d_idx` must be `rcw[d_idx]`.
                    prev_d_idx = rcw[d_idx]

                    # Length of the straight segment ending AT (pr, pc), arrived from prev_d_idx (no turn).
                    # This length is dp_straight[pr][pc][prev_d_idx] (if length >= 2) or 1 (if grid[pr][pc] == 1).
                    L_at_prpc_straight = 0
                    if 0 <= pr < n and 0 <= pc < m:
                         # Check if (pr, pc) was the start of a straight segment (length 1)
                         if grid[pr][pc] == 1:
                             L_at_prpc_straight = 1
                         # Check if (pr, pc) was the end of a straight segment length >= 2
                         elif dp_straight[pr][pc][prev_d_idx] > 0:
                             L_at_prpc_straight = dp_straight[pr][pc][prev_d_idx]


                    # If a valid straight segment ended at (pr, pc) with length L_at_prpc_straight > 0,
                    # it can turn here into direction d_idx to reach (r, c).
                    if L_at_prpc_straight > 0:
                        # The current cell (r, c) is step L_at_prpc_straight + 1 (overall step number).
                        # Required value at (r, c) for step L_at_prpc_straight + 1 is get_required_value(L_at_prpc_straight + 1).
                        # Since L_at_prpc_straight >= 1, L_at_prpc_straight + 1 >= 2.
                        required_val = get_required_value(L_at_prpc_straight + 1)

                        if grid[r][c] == required_val:
                            # This V-shaped segment ends at (r, c), arrived from d_idx, turn happened at (pr, pc).
                            # Its length is L_at_prpc_straight + 1.
                            dp_v_end[r][c][d_idx] = max(dp_v_end[r][c][d_idx], L_at_prpc_straight + 1)
                            max_length = max(max_length, dp_v_end[r][c][d_idx])


        return max_length