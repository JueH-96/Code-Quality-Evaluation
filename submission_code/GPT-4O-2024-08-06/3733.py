class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        from collections import defaultdict

        n = len(grid)
        m = len(grid[0])
        
        # Directions for diagonals: (dx, dy)
        # top-left to bottom-right
        dir1 = (1, 1)
        # bottom-right to top-left
        dir2 = (-1, -1)
        # top-right to bottom-left
        dir3 = (1, -1)
        # bottom-left to top-right
        dir4 = (-1, 1)
        
        # Function to check if a position is within bounds
        def in_bounds(x, y):
            return 0 <= x < n and 0 <= y < m
        
        # Function to follow a diagonal and return the length of the V-shaped segment
        def follow_diagonal(x, y, dx, dy, sequence):
            length = 0
            seq_index = 0
            while in_bounds(x, y) and grid[x][y] == sequence[seq_index]:
                length += 1
                seq_index = (seq_index + 1) % len(sequence)
                x += dx
                y += dy
            return length
        
        # Sequence to follow: 1, 2, 0, 2, 0, ...
        sequence = [1, 2, 0, 2, 0]
        
        max_length = 0
        
        # Check all starting points for a V-shaped diagonal
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    # Try all four diagonal directions
                    for (dx1, dy1), (dx2, dy2) in [
                        (dir1, dir3), (dir1, dir4),
                        (dir2, dir3), (dir2, dir4),
                        (dir3, dir1), (dir3, dir2),
                        (dir4, dir1), (dir4, dir2)
                    ]:
                        # First part of the diagonal
                        length1 = follow_diagonal(i, j, dx1, dy1, sequence)
                        # Second part of the diagonal after a 90-degree turn
                        if length1 > 0:
                            # Calculate the new starting point after the first diagonal
                            new_x = i + (length1 - 1) * dx1
                            new_y = j + (length1 - 1) * dy1
                            # Continue the sequence from where it left off
                            seq_index = length1 % len(sequence)
                            length2 = follow_diagonal(new_x, new_y, dx2, dy2, sequence[seq_index:])
                            max_length = max(max_length, length1 + length2)
        
        return max_length