from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        
        # Calculate the top 3 values in each row and column
        row_maxes = []
        col_maxes = []
        
        for i in range(m):
            row = sorted(board[i], reverse=True)
            row_maxes.append(row[:3])
        
        for j in range(n):
            col = sorted([board[i][j] for i in range(m)], reverse=True)
            col_maxes.append(col[:3])
        
        # Try all combinations of placing 3 rooks
        max_sum = float('-inf')
        
        for r1 in range(3):
            for c1 in range(3):
                for r2 in range(3):
                    for c2 in range(3):
                        for r3 in range(3):
                            for c3 in range(3):
                                if len({0, 1, 2} - {r1, r2, r3}) == 0 and len({0, 1, 2} - {c1, c2, c3}) == 0:
                                    max_sum = max(max_sum, row_maxes[0][r1] + row_maxes[1][r2] + row_maxes[2][r3])
        
        return max_sum