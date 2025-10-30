from typing import List
import itertools

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        max_sum = float('-inf')
        
        # Generate all possible combinations of 3 rows and 3 columns
        for rows in itertools.combinations(range(m), 3):
            for cols in itertools.combinations(range(n), 3):
                current_sum = sum(board[rows[i]][cols[i]] for i in range(3))
                max_sum = max(max_sum, current_sum)
        
        return max_sum