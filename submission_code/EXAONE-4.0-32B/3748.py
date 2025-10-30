class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        for i in range(n):
            diag = []
            r, c = i, 0
            while r < n and c < n:
                diag.append(grid[r][c])
                r += 1
                c += 1
            diag.sort(reverse=True)
            r, c = i, 0
            idx = 0
            while r < n and c < n:
                grid[r][c] = diag[idx]
                idx += 1
                r += 1
                c += 1
        
        for j in range(1, n):
            diag = []
            r, c = 0, j
            while r < n and c < n:
                diag.append(grid[r][c])
                r += 1
                c += 1
            diag.sort()
            r, c = 0, j
            idx = 0
            while r < n and c < n:
                grid[r][c] = diag[idx]
                idx += 1
                r += 1
                c += 1
        
        return grid