class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # Collect diagonals by their difference (i - j)
        diagonals = {}
        
        for i in range(n):
            for j in range(n):
                diff = i - j
                if diff not in diagonals:
                    diagonals[diff] = []
                diagonals[diff].append((i, j))
        
        # Sort each diagonal and place values back
        for diff in diagonals:
            # Extract values from current diagonal
            values = [grid[i][j] for i, j in diagonals[diff]]
            
            # Sort based on which triangle the diagonal belongs to
            if diff >= 0:  # Bottom-left triangle (including main diagonal)
                values.sort(reverse=True)  # Non-increasing order
            else:  # Top-right triangle
                values.sort()  # Non-decreasing order
            
            # Place sorted values back to their positions
            for idx, (i, j) in enumerate(diagonals[diff]):
                grid[i][j] = values[idx]
        
        return grid