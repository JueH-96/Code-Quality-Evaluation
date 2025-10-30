from typing import List

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        # d represents the difference i - j.
        # The possible values of d are from -(n-1) to (n-1)
        for d in range(-(n - 1), n):
            # Collect indices along the diagonal defined by d
            diag = []
            # i runs from max(0, d) to min(n-1, n-1+d)
            start = max(0, d)
            end = min(n - 1, n - 1 + d)
            for i in range(start, end + 1):
                j = i - d
                diag.append(grid[i][j])
            # Sort based on which triangle the diagonal belongs to:
            # if d >= 0, that diagonal is in the bottom-left triangle or main diagonal, sort descending.
            # if d < 0, that diagonal is in the top-right triangle, sort ascending.
            if d >= 0:
                diag.sort(reverse=True)
            else:
                diag.sort()
            # Write back the sorted diagonal into grid in the same order (increasing row i)
            index = 0
            for i in range(start, end + 1):
                j = i - d
                grid[i][j] = diag[index]
                index += 1
        return grid