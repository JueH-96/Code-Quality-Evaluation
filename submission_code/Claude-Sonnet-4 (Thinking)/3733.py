class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        
        # Diagonal directions: down-right, down-left, up-left, up-right
        directions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
        
        # Clockwise turn mapping
        clockwise_turn = {
            (1, 1): (1, -1),
            (1, -1): (-1, -1),
            (-1, -1): (-1, 1),
            (-1, 1): (1, 1)
        }
        
        def get_expected_value(index):
            if index == 0:
                return 1
            elif index % 2 == 1:
                return 2
            else:
                return 0
        
        def is_valid(r, c):
            return 0 <= r < n and 0 <= c < m
        
        def explore_segment(start_r, start_c, direction):
            dr, dc = direction
            max_length = 0
            
            # Explore in the initial direction and try making a turn at each position
            r, c = start_r, start_c
            length = 0
            
            while is_valid(r, c):
                expected = get_expected_value(length)
                if grid[r][c] != expected:
                    break
                length += 1
                
                # Update max_length for the current segment without turn
                max_length = max(max_length, length)
                
                # Try making a turn at this position
                new_direction = clockwise_turn[direction]
                new_dr, new_dc = new_direction
                turn_r, turn_c = r + new_dr, c + new_dc
                turn_length = length
                
                while is_valid(turn_r, turn_c):
                    expected = get_expected_value(turn_length)
                    if grid[turn_r][turn_c] != expected:
                        break
                    turn_length += 1
                    turn_r += new_dr
                    turn_c += new_dc
                
                max_length = max(max_length, turn_length)
                
                r += dr
                c += dc
            
            return max_length
        
        max_length = 0
        
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:  # Start of a potential V-shaped segment
                    for direction in directions:
                        length = explore_segment(i, j, direction)
                        max_length = max(max_length, length)
        
        return max_length