class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # Process bottom-left diagonals (starting from column 0)
        for i in range(n):
            # Collect diagonal elements
            diagonal = []
            row, col = i, 0
            while row < n and col < n:
                diagonal.append(grid[row][col])
                row += 1
                col += 1
            # Sort in non-increasing order
            diagonal.sort(reverse=True)
            # Put sorted elements back
            row, col = i, 0
            idx = 0
            while row < n and col < n:
                grid[row][col] = diagonal[idx]
                row += 1
                col += 1
                idx += 1
        
        # Process top-right diagonals (starting from row 0, column j >= 1)
        for j in range(1, n):
            # Collect diagonal elements
            diagonal = []
            row, col = 0, j
            while row < n and col < n:
                diagonal.append(grid[row][col])
                row += 1
                col += 1
            # Sort in non-decreasing order
            diagonal.sort()
            # Put sorted elements back
            row, col = 0, j
            idx = 0
            while row < n and col < n:
                grid[row][col] = diagonal[idx]
                row += 1
                col += 1
                idx += 1
        
        return grid