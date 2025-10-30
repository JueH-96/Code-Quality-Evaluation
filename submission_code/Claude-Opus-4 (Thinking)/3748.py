class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # Group elements by diagonal
        diagonals = {}
        for i in range(n):
            for j in range(n):
                key = i - j
                if key not in diagonals:
                    diagonals[key] = []
                diagonals[key].append(grid[i][j])
        
        # Sort each diagonal
        for key in diagonals:
            if key >= 0:  # Bottom-left triangle (including main diagonal)
                diagonals[key].sort(reverse=True)  # Non-increasing
            else:  # Top-right triangle
                diagonals[key].sort()  # Non-decreasing
        
        # Place sorted elements back
        diagonal_indices = {key: 0 for key in diagonals}
        for i in range(n):
            for j in range(n):
                key = i - j
                grid[i][j] = diagonals[key][diagonal_indices[key]]
                diagonal_indices[key] += 1
        
        return grid