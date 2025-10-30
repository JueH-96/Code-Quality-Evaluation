class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        # Process bottom-left diagonals (i - j = k, k >= 0)
        for k in range(n):
            elements = []
            i, j = k, 0
            while i < n and j < n:
                elements.append(grid[i][j])
                i += 1
                j += 1
            elements.sort(reverse=True)
            # Place back
            i, j = k, 0
            idx = 0
            while i < n and j < n:
                grid[i][j] = elements[idx]
                i += 1
                j += 1
                idx += 1
        
        # Process top-right diagonals (j - i = k, k >= 1)
        for k in range(1, n):
            elements = []
            i, j = 0, k
            while i < n and j < n:
                elements.append(grid[i][j])
                i += 1
                j += 1
            elements.sort()
            # Place back
            i, j = 0, k
            idx = 0
            while i < n and j < n:
                grid[i][j] = elements[idx]
                i += 1
                j += 1
                idx += 1
        
        return grid