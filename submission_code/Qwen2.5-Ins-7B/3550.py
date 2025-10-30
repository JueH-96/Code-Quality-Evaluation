from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        row_sums = [sum(row) for row in board]
        col_sums = [sum(board[i][j] for i in range(m)) for j in range(n)]
        
        max_sum = float('-inf')
        for i in range(m):
            for j in range(n):
                for k in range(m):
                    for l in range(n):
                        if i != k and j != l:
                            current_sum = board[i][j] + board[k][l]
                            if i == k:
                                current_sum += max(col_sums[j])
                            elif j == l:
                                current_sum += max(row_sums[i])
                            else:
                                current_sum += max(col_sums[j], row_sums[i])
                            max_sum = max(max_sum, current_sum)
        
        return max_sum