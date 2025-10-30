import collections

class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n = len(grid)
        m = len(grid[0])

        max_overall_len = 0

        # Define the four diagonal directions
        # dir_idx mapping:
        # 0: (1, 1)    -> Down-Right (DR)
        # 1: (1, -1)   -> Down-Left (DL)
        # 2: (-1, -1)  -> Up-Left (UL)
        # 3: (-1, 1)   -> Up-Right (UR)
        dirs = [(1, 1), (1, -1), (-1, -1), (-1, 1)]

        # dp[r][c][dir_idx][val_parity] stores the length of the straight diagonal segment
        # starting at (r, c) in direction dirs[dir_idx],
        # where grid[r][c] is expected to be:
        #   - 0 if val_parity == 0 (current value 0, next should be 2)
        #   - 2 if val_parity == 1 (current value 2, next should be 0)
        # This is for segments following the 0-2-0-2... or 2-0-2-0... patterns.
        dp = [[[[0] * 2 for _ in range(4)] for _ in range(m)] for _ in range(n)]

        # --- Precompute all straight diagonal segment lengths using DP ---
        # The iteration order depends on the direction to ensure dependencies are met.

        # dir_idx 0: (1, 1) - Down-Right. Iterate from bottom-right to top-left.
        dr, dc, dir_idx = 1, 1, 0
        for r in range(n - 1, -1, -1):
            for c in range(m - 1, -1, -1):
                # Calculate if segment starts with 0
                if grid[r][c] == 0:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < m:
                        dp[r][c][dir_idx][0] = 1 + dp[nr][nc][dir_idx][1] # Expect 2 next
                    else:
                        dp[r][c][dir_idx][0] = 1
                # Calculate if segment starts with 2
                if grid[r][c] == 2:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < m:
                        dp[r][c][dir_idx][1] = 1 + dp[nr][nc][dir_idx][0] # Expect 0 next
                    else:
                        dp[r][c][dir_idx][1] = 1

        # dir_idx 1: (1, -1) - Down-Left. Iterate from bottom-left to top-right.
        dr, dc, dir_idx = 1, -1, 1
        for r in range(n - 1, -1, -1):
            for c in range(m): # c from 0 to m-1
                if grid[r][c] == 0:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < m:
                        dp[r][c][dir_idx][0] = 1 + dp[nr][nc][dir_idx][1]
                    else:
                        dp[r][c][dir_idx][0] = 1
                if grid[r][c] == 2:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < m:
                        dp[r][c][dir_idx][1] = 1 + dp[nr][nc][dir_idx][0]
                    else:
                        dp[r][c][dir_idx][1] = 1

        # dir_idx 2: (-1, -1) - Up-Left. Iterate from top-left to bottom-right.
        dr, dc, dir_idx = -1, -1, 2
        for r in range(n): # r from 0 to n-1
            for c in range(m): # c from 0 to m-1
                if grid[r][c] == 0:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < m:
                        dp[r][c][dir_idx][0] = 1 + dp[nr][nc][dir_idx][1]
                    else:
                        dp[r][c][dir_idx][0] = 1
                if grid[r][c] == 2:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < m:
                        dp[r][c][dir_idx][1] = 1 + dp[nr][nc][dir_idx][0]
                    else:
                        dp[r][c][dir_idx][1] = 1

        # dir_idx 3: (-1, 1) - Up-Right. Iterate from top-right to bottom-left.
        dr, dc, dir_idx = -1, 1, 3
        for r in range(n): # r from 0 to n-1
            for c in range(m - 1, -1, -1): # c from m-1 to 0
                if grid[r][c] == 0:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < m:
                        dp[r][c][dir_idx][0] = 1 + dp[nr][nc][dir_idx][1]
                    else:
                        dp[r][c][dir_idx][0] = 1
                if grid[r][c] == 2:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < m:
                        dp[r][c][dir_idx][1] = 1 + dp[nr][nc][dir_idx][0]
                    else:
                        dp[r][c][dir_idx][1] = 1
        
        # --- Main loop to find the longest V-shaped diagonal segment ---
        for r_start in range(n):
            for c_start in range(m):
                if grid[r_start][c_start] == 1:
                    max_overall_len = max(max_overall_len, 1) # A single '1' is a segment of length 1

                    for initial_dir_idx in range(4):
                        dr1, dc1 = dirs[initial_dir_idx]

                        # Build the first segment (straight part before potential turn)
                        # path_segment_info stores (r, c, length_to_this_point)
                        path_segment_info = []
                        
                        r, c = r_start, c_start
                        current_len = 1
                        expected_val_for_next = 2 # After '1', expecting '2'

                        path_segment_info.append((r, c, current_len)) # Add the starting '1'

                        while True:
                            nr, nc = r + dr1, c + dc1

                            if not (0 <= nr < n and 0 <= nc < m):
                                break # Out of bounds
                            
                            if grid[nr][nc] != expected_val_for_next:
                                break # Mismatch in sequence
                            
                            current_len += 1
                            path_segment_info.append((nr, nc, current_len))
                            r, c = nr, nc
                            expected_val_for_next = 0 if expected_val_for_next == 2 else 2
                        
                        # At this point, path_segment_info contains all points of the straight first segment.
                        # Also consider the case where no turn happens (straight segment).
                        max_overall_len = max(max_overall_len, current_len)

                        # Iterate through each point in the first segment as a potential pivot point
                        for pivot_info in path_segment_info:
                            pivot_r, pivot_c, len_before_turn = pivot_info

                            # Determine the expected value for the first element *after* the pivot point
                            # in the new direction. This is based on the value *at* the pivot point.
                            val_at_pivot = grid[pivot_r][pivot_c]
                            expected_val_for_second_segment_start = -1 # Will be 0 or 2
                            if val_at_pivot == 1:
                                expected_val_for_second_segment_start = 2
                            elif val_at_pivot == 2:
                                expected_val_for_second_segment_start = 0
                            elif val_at_pivot == 0:
                                expected_val_for_second_segment_start = 2
                            
                            # Determine the new direction (90-degree clockwise turn)
                            next_dir_idx = (initial_dir_idx + 1) % 4
                            dr2, dc2 = dirs[next_dir_idx]
                            
                            current_overall_len_with_turn = len_before_turn
                            
                            # Check the first step in the new direction from the pivot point
                            r_next, c_next = pivot_r + dr2, pivot_c + dc2

                            if (0 <= r_next < n and 0 <= c_next < m) and \
                               (grid[r_next][c_next] == expected_val_for_second_segment_start):
                                
                                current_overall_len_with_turn += 1
                                
                                # Determine the 'val_parity' for the DP lookup for the rest of the segment.
                                # The DP table `dp[r][c][dir_idx][val_parity]` describes a segment
                                # *starting* at (r,c) with value 0 (if val_parity=0) or 2 (if val_parity=1).
                                # Since grid[r_next][c_next] is expected_val_for_second_segment_start,
                                # we use the corresponding val_parity.
                                next_expected_parity_for_dp_query = 0 if expected_val_for_second_segment_start == 0 else 1
                                
                                # Add the length of the remaining part of the second segment from the DP table
                                current_overall_len_with_turn += dp[r_next][c_next][next_dir_idx][next_expected_parity_for_dp_query]
                            
                            max_overall_len = max(max_overall_len, current_overall_len_with_turn)
        
        return max_overall_len