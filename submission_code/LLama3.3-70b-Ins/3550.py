from typing import List
import itertools

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        max_sum = float('-inf')
        
        # Generate all possible combinations of rows and columns
        for rows in itertools.combinations(range(m), 3):
            for cols in itertools.permutations(range(n), 3):
                # Check if the current combination is valid (no two rooks in the same row or column)
                if len(set(rows)) == 3 and len(set(cols)) == 3:
                    # Calculate the sum of the cell values for the current combination
                    current_sum = sum(board[rows[i]][cols[i]] for i in range(3))
                    # Update the maximum sum
                    max_sum = max(max_sum, current_sum)
        
        return max_sum