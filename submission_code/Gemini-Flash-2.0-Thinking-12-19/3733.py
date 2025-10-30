# @lc code=start
from typing import List

class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        dirs = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
        # dir_idx mapping: 0=(1,1), 1=(1,-1), 2=(-1,-1), 3=(-1,1)
        clockwise_turn = {0: 1, 1: 2, 2: 3, 3: 0}

        # straight_len[r][c][dir_idx][parity] stores the length of the straight segment
        # starting at (r, c) in direction dirs[dir_idx], assuming (r, c) is
        # sequence index k > 0 where k % 2 == parity (0 for even, 1 for odd).
        # The length is the number of cells in this segment starting from (r, c).
        # Initialize with -1 to indicate not computed.
        # Use -1 as sentinel for integer lengths.
        straight_len = [[[-1] * 2 for _ in range(4)] for _ in range(m)] for _ in range(n)]

        def get_straight_len(r, c, dir_idx, seq_idx_mod_2):
            if not (0 <= r < n and 0 <= c < m): return 0

            # seq_idx_mod_2 corresponds to k % 2 where k > 0
            # Required value at sequence index k (k > 0) is 2 if k is odd, 0 if k is even.
            expected_val = 2 if seq_idx_mod_2 == 1 else 0
            if grid[r][c] != expected_val: return 0

            if straight_len[r][c][dir_idx][seq_idx_mod_2] != -1:
                return straight_len[r][c][dir_idx][seq_idx_mod_2]

            dr, dc = dirs[dir_idx]
            next_r, next_c = r + dr, c + dc
            # If current is seq index k, next is k+1. Parity flips for k+1.
            # For seq indices > 0, parity of k+1 is 1 - (k%2).
            next_seq_idx_mod_2 = 1 - seq_idx_mod_2

            length = 1 + get_straight_len(next_r, next_c, dir_idx, next_seq_idx_mod_2)
            straight_len[r][c][dir_idx][seq_idx_mod_2] = length
            return length

        # Precompute by calling from all relevant states (grid value matches expected for k>0)
        # This fills the memoization table
        for r in range(n):
            for c in range(m):
                for dir_idx in range(4):
                    # Call for parity 0 (assuming cell is seq index k, k>0, k%2=0, value 0)
                    if grid[r][c] == 0:
                         get_straight_len(r, c, dir_idx, 0)
                    # Call for parity 1 (assuming cell is seq index k, k>0, k%2=1, value 2)
                    elif grid[r][c] == 2:
                         get_straight_len(r, c, dir_idx, 1)


        max_overall_len = 0

        # Iterate through all possible starting cells (r0, c0) with value 1
        for r0 in range(n):
            for c0 in range(m):
                if grid[r0][c0] == 1:
                    # Segment of length 1 is valid
                    max_overall_len = max(max_overall_len, 1)

                    # Explore V-shapes starting at (r0, c0)
                    for dir_idx1 in range(4):
                        dr1, dc1 = dirs[dir_idx1]

                        # Iterate through possible turn points (curr_r, curr_c) along the first leg
                        # curr_r, curr_c = (r0 + k*dr1, c0 + k*dc1) where k is the sequence index (k >= 1)
                        k = 1 # Sequence index of the potential turn point

                        while True:
                            curr_r, curr_c = r0 + k * dr1, c0 + k * dc1

                            if not (0 <= curr_r < n and 0 <= curr_c < m): break # Out of bounds

                            # Required value at sequence index k (k >= 1)
                            required_val_k = 2 if k % 2 == 1 else 0
                            if grid[curr_r][curr_c] != required_val_k: break # Value mismatch

                            # Found a valid cell (curr_r, curr_c) at sequence index k
                            # This cell can be a potential turning point (turn happens *at* this cell).
                            # The length of the first part including this turn cell is k + 1.

                            # Consider turning at (curr_r, curr_c) (sequence index k).
                            dir_idx2 = clockwise_turn[dir_idx1]
                            dr2, dc2 = dirs[dir_idx2]

                            # The second segment starts *after* (curr_r, curr_c) in the new direction.
                            # The first cell of the second segment is (next_r, next_c) = (curr_r + dr2, curr_c + dc2).
                            # This cell (next_r, next_c) is sequence index k + 1.
                            
                            # The length of the straight segment starting at (next_r, next_c)
                            # in direction dir_idx2, assuming (next_r, next_c) is sequence index k + 1.
                            # k >= 1, so k + 1 >= 2 > 0. Parity is (k + 1) % 2.
                            len_after_turn = get_straight_len(next_r, next_c, dir_idx2, (k + 1) % 2)

                            # Total length = (length of first part including turn cell) + (length of second part excluding turn cell)
                            # Length of first part including (curr_r, curr_c) is k + 1.
                            # Length of second part starting from (next_r, next_c) is `len_after_turn`.
                            total_len = (k + 1) + len_after_turn
                            max_overall_len = max(max_overall_len, total_len)

                            # Continue checking the straight segment (potential turn point moves one step further)
                            k += 1

        return max_overall_len

# @lc code=end