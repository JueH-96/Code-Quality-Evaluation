class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        if n == 1:
            return grid
        
        # Dictionary to hold diagonals by their starting points
        bottom_left_diagonals = {}
        top_right_diagonals = {}
        
        # Collecting values in diagonals
        for i in range(n):
            for j in range(n):
                if i >= j:  # Bottom-left triangle including the middle diagonal
                    if i - j not in bottom_left_diagonals:
                        bottom_left_diagonals[i - j] = []
                    bottom_left_diagonals[i - j].append(grid[i][j])
                if i <= j:  # Top-right triangle including the middle diagonal
                    if j - i not in top_right_diagonals:
                        top_right_diagonals[j - i] = []
                    top_right_diagonals[j - i].append(grid[i][j])
        
        # Sorting the collected diagonals
        for key in bottom_left_diagonals:
            bottom_left_diagonals[key].sort(reverse=True)
        
        for key in top_right_diagonals:
            top_right_diagonals[key].sort()
        
        # Reconstructing the grid with sorted diagonals
        result = [[0] * n for _ in range(n)]
        
        # Fill the bottom-left triangle including the middle diagonal
        for i in range(n):
            for j in range(n):
                if i >= j:  # Bottom-left triangle including the middle diagonal
                    result[i][j] = bottom_left_diagonals[i - j].pop(0)
        
        # Fill the top-right triangle
        for i in range(n):
            for j in range(n):
                if i <= j:  # Top-right triangle including the middle diagonal
                    result[i][j] = top_right_diagonals[j - i].pop(0)
        
        return result