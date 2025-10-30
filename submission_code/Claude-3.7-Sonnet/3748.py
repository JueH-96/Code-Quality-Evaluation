class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # Initialize the result matrix (deep copy of grid)
        result = [row[:] for row in grid]
        
        # Process each diagonal (diagonals are identified by the difference between row and column indices)
        for diff in range(-n+1, n):
            diagonal = []
            positions = []
            
            # Get elements and positions of the current diagonal
            for i in range(n):
                j = i - diff
                if 0 <= j < n:
                    diagonal.append(grid[i][j])
                    positions.append((i, j))
            
            # Sort the diagonal (only if it has more than one element)
            if len(diagonal) > 1:
                if diff >= 0:  # Bottom-left triangle (including the main diagonal)
                    diagonal.sort(reverse=True)  # Non-increasing order
                else:  # Top-right triangle
                    diagonal.sort()  # Non-decreasing order
            
            # Update the result matrix
            for (i, j), val in zip(positions, diagonal):
                result[i][j] = val
        
        return result