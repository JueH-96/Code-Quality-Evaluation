import itertools
from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        cells = []
        for i in range(m):
            for j in range(n):
                cells.append((board[i][j], i, j))
        # Sort cells in descending order of their value
        cells.sort(reverse=True, key=lambda x: x[0])
        K = 3 * max(m, n)
        top_cells = cells[:K]
        max_sum = -float('inf')
        # Check all triplets in top_cells
        for triplet in itertools.combinations(top_cells, 3):
            rows = set()
            cols = set()
            valid = True
            sum_val = 0
            for cell in triplet:
                val, r, c = cell
                if r in rows or c in cols:
                    valid = False
                    break
                rows.add(r)
                cols.add(c)
                sum_val += val
            if valid:
                if sum_val > max_sum:
                    max_sum = sum_val
        return max_sum