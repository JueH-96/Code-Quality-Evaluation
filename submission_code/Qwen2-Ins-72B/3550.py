from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        row_sums = [sum(x != 0 for x in row) for row in board]
        col_sums = [sum(row[i] != 0 for row in board) for i in range(n)]
        
        def is_valid(i, j):
            return row_sums[i] < 2 and col_sums[j] < 2
        
        max_sum = float('-inf')
        for i in range(m):
            for j in range(n):
                if is_valid(i, j):
                    row_sums[i] += 1
                    col_sums[j] += 1
                    for k in range(m):
                        if k != i and is_valid(k, j):
                            for l in range(n):
                                if l != j and is_valid(i, l) and is_valid(k, l):
                                    max_sum = max(max_sum, board[i][j] + board[k][j] + board[i][l])
                    row_sums[i] -= 1
                    col_sums[j] -= 1
        
        return max_sum if max_sum != float('-inf') else 0