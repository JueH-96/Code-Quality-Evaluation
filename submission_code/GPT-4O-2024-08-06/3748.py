class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # Function to extract a diagonal
        def extract_diagonal(r, c):
            diagonal = []
            while r < n and c < n:
                diagonal.append(grid[r][c])
                r += 1
                c += 1
            return diagonal
        
        # Function to insert a sorted diagonal back into the grid
        def insert_diagonal(r, c, diagonal):
            i = 0
            while r < n and c < n:
                grid[r][c] = diagonal[i]
                r += 1
                c += 1
                i += 1
        
        # Sort the diagonals in the bottom-left triangle (including the middle diagonal)
        for start_row in range(n):
            diagonal = extract_diagonal(start_row, 0)
            diagonal.sort(reverse=True)
            insert_diagonal(start_row, 0, diagonal)
        
        # Sort the diagonals in the top-right triangle
        for start_col in range(1, n):
            diagonal = extract_diagonal(0, start_col)
            diagonal.sort()
            insert_diagonal(0, start_col, diagonal)
        
        return grid