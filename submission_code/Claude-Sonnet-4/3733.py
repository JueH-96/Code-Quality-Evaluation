class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        
        # Define the 4 diagonal directions: down-right, down-left, up-left, up-right
        directions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
        
        # Define clockwise 90-degree turns for each direction
        clockwise_turn = {
            (1, 1): (1, -1),    # down-right to down-left
            (1, -1): (-1, -1),  # down-left to up-left
            (-1, -1): (-1, 1),  # up-left to up-right
            (-1, 1): (1, 1)     # up-right to down-right
        }
        
        def get_expected_value(pos):
            # Pattern: 1, 2, 0, 2, 0, 2, 0, ...
            if pos == 0:
                return 1
            elif pos == 1:
                return 2
            else:
                return 0 if pos % 2 == 0 else 2
        
        def is_valid(r, c):
            return 0 <= r < n and 0 <= c < m
        
        def dfs(start_r, start_c, direction, pos, turned):
            r, c = start_r, start_c
            dr, dc = direction
            
            # Check if current position matches expected value
            if not is_valid(r, c) or grid[r][c] != get_expected_value(pos):
                return 0
            
            max_len = 1
            
            # Try continuing in the same direction
            next_r, next_c = r + dr, c + dc
            if is_valid(next_r, next_c):
                max_len = max(max_len, 1 + dfs(next_r, next_c, direction, pos + 1, turned))
            
            # Try making a clockwise turn (only if we haven't turned yet)
            if not turned:
                new_direction = clockwise_turn[direction]
                new_dr, new_dc = new_direction
                turn_r, turn_c = r + new_dr, c + new_dc
                if is_valid(turn_r, turn_c):
                    max_len = max(max_len, 1 + dfs(turn_r, turn_c, new_direction, pos + 1, True))
            
            return max_len
        
        max_length = 0
        
        # Try starting from each cell with value 1
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    # Try all 4 diagonal directions
                    for direction in directions:
                        length = dfs(i, j, direction, 0, False)
                        max_length = max(max_length, length)
        
        return max_length