class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)

        # Sort bottom-left diagonals (non-increasing)
        for k in range(1 - n, 0):
            diag = []
            for i in range(n):
                j = i - k
                if 0 <= i < n and 0 <= j < n:
                    diag.append(grid[i][j])
            diag.sort(reverse=True)
            idx = 0
            for i in range(n):
                j = i - k
                if 0 <= i < n and 0 <= j < n:
                    grid[i][j] = diag[idx]
                    idx += 1

        # Sort top-right diagonals (non-decreasing)
        for k in range(0, n - 1):
            diag = []
            for i in range(n):
                j = i + k + 1
                if 0 <= i < n and 0 <= j < n:
                    diag.append(grid[i][j])
            diag.sort()
            idx = 0
            for i in range(n):
                j = i + k + 1
                if 0 <= i < n and 0 <= j < n:
                    grid[i][j] = diag[idx]
                    idx += 1

        return grid