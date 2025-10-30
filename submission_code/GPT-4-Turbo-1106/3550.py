from typing import List
from itertools import permutations

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        max_sum = float('-inf')
        
        # Generate all possible combinations of 3 distinct rows and 3 distinct columns
        for rows in permutations(range(m), 3):
            for cols in permutations(range(n), 3):
                # Calculate the sum of the cell values for the current combination
                current_sum = board[rows[0]][cols[0]] + board[rows[1]][cols[1]] + board[rows[2]][cols[2]]
                max_sum = max(max_sum, current_sum)
        
        return max_sum