class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        
        # Define diagonal directions: NE, SE, SW, NW
        directions = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
        
        # Expected sequence: 1, 2, 0, 2, 0, 2, 0, ...
        def get_expected_value(index):
            if index == 0:
                return 1
            elif index == 1:
                return 2
            else:
                return 0 if index % 2 == 0 else 2
        
        # Check if a position is valid
        def is_valid(r, c):
            return 0 <= r < n and 0 <= c < m
        
        # Get the length of a V-shaped segment starting from (r, c)
        def get_v_length(start_r, start_c, initial_dir):
            if grid[start_r][start_c] != 1:
                return 0
            
            max_length = 0
            
            # Try without turn and with turn
            for use_turn in [False, True]:
                r, c = start_r, start_c
                length = 1
                current_dir = initial_dir
                turned = False
                
                while True:
                    # Get next position
                    dr, dc = directions[current_dir]
                    next_r, next_c = r + dr, c + dc
                    
                    # Check if we can continue in current direction
                    if is_valid(next_r, next_c) and grid[next_r][next_c] == get_expected_value(length):
                        r, c = next_r, next_c
                        length += 1
                    elif use_turn and not turned:
                        # Try to turn 90 degrees clockwise
                        turned = True
                        new_dir = (current_dir + 1) % 4
                        dr, dc = directions[new_dir]
                        next_r, next_c = r + dr, c + dc
                        
                        if is_valid(next_r, next_c) and grid[next_r][next_c] == get_expected_value(length):
                            r, c = next_r, next_c
                            length += 1
                            current_dir = new_dir
                        else:
                            break
                    else:
                        break
                
                max_length = max(max_length, length)
            
            return max_length
        
        # Find the maximum V-shaped segment
        result = 0
        
        # Check all positions that have value 1
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    # Try all four initial directions
                    for dir_idx in range(4):
                        result = max(result, get_v_length(i, j, dir_idx))
        
        return result