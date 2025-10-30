from typing import List

class Solution:
    def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        
        # Initialize the result matrix with zeros.
        # This will hold the sorted elements.
        res = [[0] * n for _ in range(n)]

        # Iterate through all possible diagonal "indices" (i - j).
        # These indices `k` range from -(n-1) to (n-1).
        # For example, if n=3, k ranges from -2 (for grid[0][2]) to 2 (for grid[2][0]).
        # The range function `range(start, stop)` goes up to `stop - 1`, so `n` is correct for `stop`.
        for k in range(-(n - 1), n):
            current_diagonal_elements = []
            
            # Determine the starting coordinates (r_start, c_start) for the current diagonal.
            # A diagonal defined by `i - j = k` starts:
            # - at row `k` and column `0` if `k` is non-negative (i.e., on or below the main diagonal).
            # - at row `0` and column `-k` if `k` is negative (i.e., above the main diagonal).
            r_start, c_start = 0, 0
            if k >= 0:
                r_start = k
                c_start = 0
            else: # k < 0
                r_start = 0
                c_start = -k
            
            # Collect all elements along the current diagonal.
            # We traverse the diagonal by simultaneously incrementing both row and column indices.
            r, c = r_start, c_start
            while 0 <= r < n and 0 <= c < n:
                current_diagonal_elements.append(grid[r][c])
                r += 1
                c += 1
            
            # Sort the collected elements based on the problem's rules.
            # Diagonals with k >= 0 (which are the main diagonal and those below it)
            # must be sorted in non-increasing (descending) order.
            # Diagonals with k < 0 (which are those above the main diagonal)
            # must be sorted in non-decreasing (ascending) order.
            if k >= 0: 
                current_diagonal_elements.sort(reverse=True) # Sort in descending order
            else: 
                current_diagonal_elements.sort() # Sort in ascending order
            
            # Place the sorted elements back into their respective positions in the result matrix.
            # We use the same traversal logic (starting from r_start, c_start and incrementing r, c)
            # as when collecting elements to ensure that the sorted values are placed
            # into the correct original cell positions along the diagonal.
            element_idx = 0
            r, c = r_start, c_start
            while 0 <= r < n and 0 <= c < n:
                res[r][c] = current_diagonal_elements[element_idx]
                element_idx += 1
                r += 1
                c += 1
        
        return res