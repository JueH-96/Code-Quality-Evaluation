from typing import List

class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])
        dirs = [(1, 1), (1, -1), (-1, -1), (-1, 1)] # TL-BR, TR-BL, BR-TL, BL-TR

        # required_val_k(k): Value required at overall segment index k (0-indexed).
        # k=0: 1
        # k>0, k odd: 2
        # k>0, k even: 0
        def required_val_k(k):
            if k == 0:
                return 1
            elif k % 2 == 1:
                return 2
            else: # k > 0 and k % 2 == 0
                return 0

        # required_relative_val_idx(j, start_val_idx):
        # Value required at relative index j (0-indexed) in a sequence starting at (r,c).
        # This sequence starts with a value based on start_val_idx.
        # start_val_idx 0: sequence starts with 0, then 2, 0, 2... (0, 2, 0, 2, ...)
        # start_val_idx 1: sequence starts with 2, then 0, 2, 0... (2, 0, 2, 0, ...)
        def required_relative_val_idx(j, start_val_idx):
            if start_val_idx == 1: # Starts with 2 (seq 2,0,2,0...)
                return 2 if j % 2 == 0 else 0 # j=0->2, j=1->0, j=2->2...
            elif start_val_idx == 0: # Starts with 0 (seq 0,2,0,2...)
                return 0 if j % 2 == 0 else 2 # j=0->0, j=1->2, j=2->0...
            return -1 # Should not happen

        # straight_len[r][c][dir_idx]: length of straight segment ending at (r,c) in dir_idx, following 1, 2, 0, ... sequence from the beginning.
        straight_len = [[[0 for _ in range(4)] for _ in range(m)] for _ in range(n)]

        # post_turn_len[dir_idx][r][c][start_val_idx]: length of straight segment starting at (r,c) in dir_idx, following sequence based on the value expected at (r,c) itself relative to the start of this segment part.
        # start_val_idx 0 means (r,c) is expected to be 0, then seq 0, 2, 0, 2...
        # start_val_idx 1 means (r,c) is expected to be 2, then seq 2, 0, 2, 0...
        # The DP state is [dir_idx][r][c][start_val_idx] for consistency with iteration order.
        post_turn_len = [[[[0 for _ in range(2)] for _ in range(m)] for _ in range(n)] for _ in range(4)]

        # Compute straight_len
        for dir_idx in range(4):
            dr, dc = dirs[dir_idx]
            r_range = range(n) if dr > 0 else range(n - 1, -1, -1)
            c_range = range(m) if dc > 0 else range(m - 1, -1, -1)

            for r in r_range:
                for c in c_range:
                    if grid[r][c] == 1:
                        straight_len[r][c][dir_idx] = 1
                    else:
                        pr, pc = r - dr, c - dc
                        if 0 <= pr < n and 0 <= pc < m and straight_len[pr][pc][dir_idx] > 0:
                            k_prev = straight_len[pr][pc][dir_idx] # Length ending at previous cell (pr, pc)
                            k_curr = k_prev + 1 # Length ending at current cell (r, c)
                            
                            # The current cell (r, c) is at overall segment index k_curr - 1.
                            # Value at overall index k (k_curr-1) must be required_val_k(k_curr-1).
                            required_val = required_val_k(k_curr - 1)

                            if grid[r][c] == required_val:
                                straight_len[r][c][dir_idx] = k_curr

        # Compute post_turn_len
        # Iterate in reverse direction for each diagonal
        # dir 0 (1, 1): r n-1..0, c m-1..0
        # dir 1 (1, -1): r n-1..0, c 0..m-1
        # dir 2 (-1, -1): r 0..n-1, c 0..m-1
        # dir 3 (-1, 1): r 0..n-1, c m-1..0
        dir_reverse_ranges = [
            (range(n - 1, -1, -1), range(m - 1, -1, -1)), # Dir 0 (1,1) reverse
            (range(n - 1, -1, -1), range(m)),             # Dir 1 (1,-1) reverse
            (range(n), range(m)),                         # Dir 2 (-1,-1) reverse
            (range(n), range(m - 1, -1, -1)),             # Dir 3 (-1,1) reverse
        ]

        for dir_idx in range(4):
            dr, dc = dirs[dir_idx]
            r_range, c_range = dir_reverse_ranges[dir_idx]

            for r in r_range:
                for c in c_range:
                    # start_val_idx 0: Sequence starts with 0 (expected at relative index j=0)
                    # start_val_idx 1: Sequence starts with 2 (expected at relative index j=0)
                    for start_val_idx in range(2):
                        # (r, c) is the starting point for this segment part (relative index j=0)
                        required_curr_at_j0 = required_relative_val_idx(0, start_val_idx)
                        if grid[r][c] == required_curr_at_j0:
                             post_turn_len[dir_idx][r][c][start_val_idx] = 1

                             # Check the next cell (r+dr, c+dc) which is relative index j=1
                             nr, nc = r + dr, c + dc
                             if 0 <= nr < n and 0 <= nc < m:
                                 # The value at (nr, nc) (relative index j=1 from (r,c)) must match required_relative_val_idx(1, start_val_idx).
                                 # This value determines the required starting value for the segment beginning *from* (nr, nc) (relative index j'=0 from (nr,nc)).
                                 next_start_val_at_nrnc = required_relative_val_idx(1, start_val_idx)

                                 # Find the start_val_idx corresponding to next_start_val_at_nrnc (0 if 0, 1 if 2)
                                 next_start_val_idx_for_nrnc = 0 if next_start_val_at_nrnc == 0 else 1

                                 if post_turn_len[dir_idx][nr][nc][next_start_val_idx_for_nrnc] > 0:
                                    post_turn_len[dir_idx][r][c][start_val_idx] = 1 + post_turn_len[dir_idx][nr][nc][next_start_val_idx_for_nrnc]

        # Mapping from dir_idx to clockwise 90-degree turned dir_idx
        turn_map = {0: 1, 1: 2, 2: 3, 3: 0}

        max_len = 0

        # Max length from straight segments (0 turns)
        for r in range(n):
            for c in range(m):
                for dir_idx in range(4):
                    max_len = max(max_len, straight_len[r][c][dir_idx])

        # Max length from segments with one turn
        # Iterate over all potential turn points (tr, tc)
        for tr in range(n):
            for tc in range(m):
                # Iterate over all possible incoming directions (prev_dir_idx)
                for prev_dir_idx in range(4):
                    L1 = straight_len[tr][tc][prev_dir_idx]

                    if L1 >= 1: # Must have a valid first part ending at (tr, tc)
                        curr_dir_idx = turn_map[prev_dir_idx]

                        if L1 == 1:
                            # Turn at overall index 0. (tr, tc) is the starting cell (value 1).
                            # The segment is a straight line starting at (tr, tc) in curr_dir, following 1, 2, 0, ...
                            # The length is given by straight_len[tr][tc][curr_dir_idx].
                            # Note: if L1=1, grid[tr][tc] == 1. straight_len calculation for curr_dir_idx starting at (tr,tc) will correctly start with length 1.
                             max_len = max(max_len, straight_len[tr][tc][curr_dir_idx])
                        else: # L1 > 1. Turn happens at overall index L1-1.
                            # (tr, tc) is at overall index L1-1.
                            # The second part starts at (tr, tc) in curr_dir.
                            # The sequence of values required for the second part, starting at (tr, tc), is dictated by the overall indices L1-1, L1, L1+1, ...
                            # required values: required_val_k(L1-1), required_val_k(L1), required_val_k(L1+1), ...

                            # The starting value for the post_turn_len DP at (tr, tc) (relative index 0 in second part) is required_val_k(L1-1).
                            # Since L1 > 1, L1-1 >= 1.
                            required_start_val_at_trtc = required_val_k(L1 - 1)

                            # Find the start_val_idx for post_turn_len corresponding to required_start_val_at_trtc.
                            # This corresponds to using post_turn_len[...][0] if the sequence starts with 0, and post_turn_len[...][1] if it starts with 2.
                            required_start_val_idx_for_trtc = 0 if required_start_val_at_trtc == 0 else 1

                            L2 = post_turn_len[curr_dir_idx][tr][tc][required_start_val_idx_for_trtc]

                            if L2 > 0: # The second part must have length >= 1
                                # Total length = Length of first part (L1) + Length of second part (L2) - 1 (overlap at (tr, tc))
                                total_len = L1 + L2 - 1
                                max_len = max(max_len, total_len)

        return max_len