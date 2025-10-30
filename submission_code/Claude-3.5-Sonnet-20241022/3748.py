class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # Helper function to get diagonal elements
        def get_diagonal(start_row, start_col):
            elements = []
            r, c = start_row, start_col
            while r < n and c < n:
                elements.append(grid[r][c])
                r += 1
                c += 1
            return elements
        
        # Helper function to set diagonal elements
        def set_diagonal(start_row, start_col, elements):
            r, c = start_row, start_col
            i = 0
            while r < n and c < n:
                grid[r][c] = elements[i]
                r += 1
                c += 1
                i += 1
        
        # Sort bottom-left triangle (including middle diagonal)
        for i in range(n):
            diagonal = get_diagonal(i, 0)
            diagonal.sort(reverse=True)  # Sort in non-increasing order
            set_diagonal(i, 0, diagonal)
        
        # Sort top-right triangle (excluding middle diagonal)
        for j in range(1, n):
            diagonal = get_diagonal(0, j)
            diagonal.sort()  # Sort in non-decreasing order
            set_diagonal(0, j, diagonal)
        
        return grid