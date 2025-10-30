class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        
        # Directions: Down-right, Down-left, Up-left, Up-right (in clockwise order)
        directions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
        
        # Determine expected value at a given sequence index
        def expected_val(idx):
            if idx == 0:
                return 1
            elif idx % 2 == 1:
                return 2
            else:
                return 0
        
        # Trace sequence in a specific direction starting from a position
        def trace_sequence(i, j, dir_idx, seq_idx):
            length = 0
            
            while 0 <= i < n and 0 <= j < m and grid[i][j] == expected_val(seq_idx):
                length += 1
                di, dj = directions[dir_idx]
                i, j = i + di, j + dj
                seq_idx += 1
            
            return length
        
        max_length = 0
        
        # Try each cell with value 1 as a starting point
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    # Try each of the four diagonal directions
                    for dir_idx in range(4):
                        di, dj = directions[dir_idx]
                        curr_i, curr_j = i, j
                        curr_seq_idx = 0
                        curr_length = 0
                        
                        # Trace the diagonal without turning first
                        while 0 <= curr_i < n and 0 <= curr_j < m and grid[curr_i][curr_j] == expected_val(curr_seq_idx):
                            curr_length += 1
                            
                            # Try making a turn at this point
                            turn_dir_idx = (dir_idx + 1) % 4  # 90-degree clockwise turn
                            turn_di, turn_dj = directions[turn_dir_idx]
                            turn_i, turn_j = curr_i + turn_di, curr_j + turn_dj
                            turn_seq_idx = curr_seq_idx + 1
                            
                            # Trace after the turn
                            turn_length = trace_sequence(turn_i, turn_j, turn_dir_idx, turn_seq_idx)
                            
                            # Update max length if we found a longer V-shaped segment
                            max_length = max(max_length, curr_length + turn_length)
                            
                            # Continue in original direction
                            curr_i += di
                            curr_j += dj
                            curr_seq_idx += 1
                        
                        # Also consider not turning at all
                        max_length = max(max_length, curr_length)
        
        return max_length