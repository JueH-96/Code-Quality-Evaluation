class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # Process all diagonals
        # Diagonals can be identified by (row - col) value
        # For each unique (row - col), we get one diagonal
        
        diagonals = {}
        
        # Collect all diagonals
        for i in range(n):
            for j in range(n):
                diff = i - j
                if diff not in diagonals:
                    diagonals[diff] = []
                diagonals[diff].append((grid[i][j], i, j))
        
        # Sort each diagonal
        for diff in diagonals:
            values_and_positions = diagonals[diff]
            values = [x[0] for x in values_and_positions]
            
            if diff >= 0:  # Bottom-left triangle (including main diagonal)
                values.sort(reverse=True)  # Non-increasing
            else:  # Top-right triangle
                values.sort()  # Non-decreasing
            
            # Put sorted values back
            for i, (_, row, col) in enumerate(values_and_positions):
                grid[row][col] = values[i]
        
        return grid