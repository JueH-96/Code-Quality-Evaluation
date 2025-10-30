from itertools import combinations
from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        max_sum = -float('inf')
        
        # Precompute top 3 values for each row and column
        top_rows = []
        for row in board:
            sorted_row = sorted([(val, col) for col, val in enumerate(row)], key=lambda x: (-x[0], x[1]))
            top_rows.append(sorted_row[:3])
        
        top_cols = []
        for col in range(n):
            column = [board[row][col] for row in range(m)]
            sorted_col = sorted([(val, row) for row, val in enumerate(column)], key=lambda x: (-x[0], x[1]))
            top_cols.append(sorted_col[:3])
        
        # Check all combinations of 3 rows and their top columns
        for rows in combinations(range(m), 3):
            i, j, k = rows
            for a in range(len(top_rows[i])):
                val_i, col_i = top_rows[i][a]
                for b in range(len(top_rows[j])):
                    val_j, col_j = top_rows[j][b]
                    for c in range(len(top_rows[k])):
                        val_k, col_k = top_rows[k][c]
                        if col_i != col_j and col_j != col_k and col_i != col_k:
                            current_sum = val_i + val_j + val_k
                            if current_sum > max_sum:
                                max_sum = current_sum
        
        # Check all combinations of 3 columns and their top rows
        for cols in combinations(range(n), 3):
            c1, c2, c3 = cols
            for a in range(len(top_cols[c1])):
                val_a, row_a = top_cols[c1][a]
                for b in range(len(top_cols[c2])):
                    val_b, row_b = top_cols[c2][b]
                    for c in range(len(top_cols[c3])):
                        val_c, row_c = top_cols[c3][c]
                        if row_a != row_b and row_b != row_c and row_a != row_c:
                            current_sum = val_a + val_b + val_c
                            if current_sum > max_sum:
                                max_sum = current_sum
        
        return max_sum