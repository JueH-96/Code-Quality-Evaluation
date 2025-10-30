from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        
        max_sum = float('-inf')

        for r1 in range(m):
            for c1 in range(n):
                for r2 in range(m):
                    if r2 == r1:
                        continue
                    for c2 in range(n):
                        if c2 == c1:
                            continue
                        for r3 in range(m):
                            if r3 == r1 or r3 == r2:
                                continue
                            for c3 in range(n):
                                if c3 == c1 or c3 == c2:
                                    continue
                                current_sum = board[r1][c1] + board[r2][c2] + board[r3][c3]
                                max_sum = max(max_sum, current_sum)
        
        return max_sum