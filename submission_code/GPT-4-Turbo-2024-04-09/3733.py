class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        
        # Directions: (dx, dy)
        # 0: top-left to bottom-right
        # 1: top-right to bottom-left
        # 2: bottom-left to top-right
        # 3: bottom-right to top-left
        directions = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        
        def is_valid(x, y):
            return 0 <= x < n and 0 <= y < m
        
        def follow_direction(x, y, direction, expected_value):
            length = 0
            dx, dy = directions[direction]
            while is_valid(x, y) and grid[x][y] == expected_value:
                length += 1
                expected_value = 2 if expected_value == 0 else 0
                x += dx
                y += dy
            return length
        
        max_length = 0
        
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    # Start from each 1 and try all directions and their 90-degree clockwise turns
                    for d in range(4):
                        # Initial direction
                        initial_length = follow_direction(i, j, d, 1)
                        if initial_length > 1:
                            # Turn 90 degrees clockwise
                            turn_direction = (d + 1) % 4
                            # Last valid position in initial direction
                            last_x = i + (initial_length - 1) * directions[d][0]
                            last_y = j + (initial_length - 1) * directions[d][1]
                            # Expected next value after initial sequence
                            next_expected_value = 2 if (initial_length - 1) % 2 == 0 else 0
                            # Continue in the turned direction
                            turned_length = follow_direction(last_x, last_y, turn_direction, next_expected_value)
                            total_length = initial_length + turned_length - 1  # Subtract the turn point counted twice
                        else:
                            total_length = initial_length
                        
                        max_length = max(max_length, total_length)
        
        return max_length