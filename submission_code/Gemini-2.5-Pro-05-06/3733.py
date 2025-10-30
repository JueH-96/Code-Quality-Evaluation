import collections

class Solution:
    def lenOfVDiagonal(self, grid: list[list[int]]) -> int:
        n = len(grid)
        m = len(grid[0])

        memo_expected_val = {}
        def get_expected_val(seq_type: int, length_idx: int) -> int:
            state = (seq_type, length_idx)
            if state in memo_expected_val:
                return memo_expected_val[state]

            if seq_type == 0: # 1, 2, 0, ...
                if length_idx == 0: result = 1
                elif length_idx % 2 == 1: result = 2
                else: result = 0
            elif seq_type == 1: # 2, 0, ...
                if length_idx % 2 == 0: result = 2
                else: result = 0
            else: # seq_type == 2: 0, 2, ...
                if length_idx % 2 == 0: result = 0
                else: result = 2
            
            memo_expected_val[state] = result
            return result

        dp_ending_at = [[[([0] * m) for _ in range(n)] for _ in range(4)] for _ in range(3)]
        dp_starting_at = [[[([0] * m) for _ in range(n)] for _ in range(4)] for _ in range(3)]

        path_dirs = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
        turn_map = {0: 2, 1: 3, 2: 1, 3: 0}
        max_overall_len = 0

        iter_configs_ending = [
            (range(n), range(m)),
            (range(n - 1, -1, -1), range(m - 1, -1, -1)),
            (range(n), range(m - 1, -1, -1)),
            (range(n - 1, -1, -1), range(m)),
        ]
        
        for dir_idx in range(4):
            dr, dc = path_dirs[dir_idx]
            r_iter, c_iter_template = iter_configs_ending[dir_idx]
            
            for r in r_iter:
                c_iter = list(c_iter_template) # Ensure fresh iterator if template is a generator (list() handles ranges too)
                for c in c_iter:
                    pr, pc = r - dr, c - dc
                    
                    # Seq type 0 (1,2,0,...)
                    if grid[r][c] == get_expected_val(0, 0):
                        dp_ending_at[0][dir_idx][r][c] = 1
                    elif 0 <= pr < n and 0 <= pc < m:
                        prev_len = dp_ending_at[0][dir_idx][pr][pc]
                        if prev_len > 0 and grid[r][c] == get_expected_val(0, prev_len):
                            dp_ending_at[0][dir_idx][r][c] = prev_len + 1
                    max_overall_len = max(max_overall_len, dp_ending_at[0][dir_idx][r][c])

                    # Seq type 1 (2,0,...)
                    if grid[r][c] == get_expected_val(1,0): # Current is 2
                        val = 1
                        if 0 <= pr < n and 0 <= pc < m and grid[pr][pc] == get_expected_val(2,0): # Prev is 0
                            if dp_ending_at[2][dir_idx][pr][pc] > 0:
                                val = 1 + dp_ending_at[2][dir_idx][pr][pc]
                        dp_ending_at[1][dir_idx][r][c] = val
                    
                    # Seq type 2 (0,2,...)
                    if grid[r][c] == get_expected_val(2,0): # Current is 0
                        val = 1
                        if 0 <= pr < n and 0 <= pc < m and grid[pr][pc] == get_expected_val(1,0): # Prev is 2
                            if dp_ending_at[1][dir_idx][pr][pc] > 0:
                                val = 1 + dp_ending_at[1][dir_idx][pr][pc]
                        dp_ending_at[2][dir_idx][r][c] = val
        
        iter_configs_starting = [
            (range(n - 1, -1, -1), range(m - 1, -1, -1)),
            (range(n), range(m)),
            (range(n - 1, -1, -1), range(m)),
            (range(n), range(m-1, -1, -1)),
        ]

        for dir_idx in range(4):
            dr, dc = path_dirs[dir_idx]
            r_iter, c_iter_template = iter_configs_starting[dir_idx]

            for r in r_iter:
                c_iter = list(c_iter_template)
                for c in c_iter:
                    nr, nc = r + dr, c + dc

                    # Seq type 0 (1,2,0,...)
                    if grid[r][c] == get_expected_val(0, 0): # Current is 1
                        val = 1
                        if 0 <= nr < n and 0 <= nc < m and grid[nr][nc] == get_expected_val(1,0): # Next is 2
                            if dp_starting_at[1][dir_idx][nr][nc] > 0:
                                val = 1 + dp_starting_at[1][dir_idx][nr][nc]
                        dp_starting_at[0][dir_idx][r][c] = val
                    
                    # Seq type 1 (2,0,...)
                    if grid[r][c] == get_expected_val(1,0): # Current is 2
                        val = 1
                        if 0 <= nr < n and 0 <= nc < m and grid[nr][nc] == get_expected_val(2,0): # Next is 0
                            if dp_starting_at[2][dir_idx][nr][nc] > 0:
                                val = 1 + dp_starting_at[2][dir_idx][nr][nc]
                        dp_starting_at[1][dir_idx][r][c] = val

                    # Seq type 2 (0,2,...)
                    if grid[r][c] == get_expected_val(2,0): # Current is 0
                        val = 1
                        if 0 <= nr < n and 0 <= nc < m and grid[nr][nc] == get_expected_val(1,0): # Next is 2
                            if dp_starting_at[1][dir_idx][nr][nc] > 0:
                                val = 1 + dp_starting_at[1][dir_idx][nr][nc]
                        dp_starting_at[2][dir_idx][r][c] = val
        
        for r_turn in range(n):
            for c_turn in range(m):
                for in_path_idx in range(4):
                    L1 = dp_ending_at[0][in_path_idx][r_turn][c_turn]
                    if L1 == 0:
                        continue

                    out_path_idx = turn_map[in_path_idx]
                    dr_out, dc_out = path_dirs[out_path_idx]
                    
                    r_next_arm2, c_next_arm2 = r_turn + dr_out, c_turn + dc_out
                    L2 = 0
                    if 0 <= r_next_arm2 < n and 0 <= c_next_arm2 < m:
                        expected_val_at_L1 = get_expected_val(0, L1) # Element value for index L1 in type 0 seq
                        
                        # This value should start arm 2. Test if it's 2 or 0.
                        if expected_val_at_L1 == get_expected_val(1,0): # Expected start of arm2 is 2
                           if grid[r_next_arm2][c_next_arm2] == get_expected_val(1,0):
                               L2 = dp_starting_at[1][out_path_idx][r_next_arm2][c_next_arm2]
                        elif expected_val_at_L1 == get_expected_val(2,0): # Expected start of arm2 is 0
                           if grid[r_next_arm2][c_next_arm2] == get_expected_val(2,0):
                               L2 = dp_starting_at[2][out_path_idx][r_next_arm2][c_next_arm2]
                    
                    if L2 > 0 :
                        max_overall_len = max(max_overall_len, L1 + L2)
        
        return max_overall_len