class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        max_length = 0
        
        # Directions for diagonal movements: (dx, dy)
        directions = [
            (1, 1),   # Down-Right
            (-1, -1), # Up-Left
            (1, -1),  # Down-Left
            (-1, 1)   # Up-Right
        ]
        
        # Function to check if the next element in the sequence is valid
        def is_valid(x, y, expected):
            return 0 <= x < n and 0 <= y < m and grid[x][y] == expected
        
        # Check for V-shaped segments starting from each cell
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    # Try each direction
                    for d1 in range(4):
                        x, y = i, j
                        length = 1
                        # First diagonal direction
                        while True:
                            x += directions[d1][0]
                            y += directions[d1][1]
                            if is_valid(x, y, 2):
                                length += 1
                            else:
                                break
                            
                            # Now try to turn 90 degrees clockwise
                            turn_direction = (d1 + 1) % 4
                            x2, y2 = x, y
                            while True:
                                x2 += directions[turn_direction][0]
                                y2 += directions[turn_direction][1]
                                if is_valid(x2, y2, 0):
                                    length += 1
                                else:
                                    break
                                
                                # Check if we can continue in the same direction
                                if is_valid(x2 + directions[turn_direction][0], y2 + directions[turn_direction][1], 2):
                                    length += 1
                                    break
                            
                            max_length = max(max_length, length)
        
        return max_length