from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        max_sum = float('-inf')
        
        for i in range(m):
            for j in range(n):
                for k in range(i+1, m):
                    for l in range(n):
                        if l == j:
                            continue
                        for p in range(k+1, m):
                            for q in range(n):
                                if q == j or q == l:
                                    continue
                                current_sum = board[i][j] + board[k][l] + board[p][q]
                                max_sum = max(max_sum, current_sum)
        
        return max_sum