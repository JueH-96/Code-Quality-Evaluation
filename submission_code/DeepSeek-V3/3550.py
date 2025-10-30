from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        max_sum = -float('inf')
        
        # Iterate over all possible combinations of 3 rows
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    # For each combination of 3 rows, find the maximum sum of 3 cells
                    # such that no two are in the same column
                    # We need to select 3 different columns
                    # So we need to find the top 3 values in the selected rows, ensuring no column conflict
                    # To do this, we can iterate over all possible column combinations
                    # Since n is up to 100, iterating over all combinations is not feasible
                    # Instead, we can precompute the top 3 values for each row and then find a combination
                    # that does not have column conflicts
                    
                    # Precompute the top 3 values for each row
                    row1 = sorted([(board[i][col], col) for col in range(n)], reverse=True)[:3]
                    row2 = sorted([(board[j][col], col) for col in range(n)], reverse=True)[:3]
                    row3 = sorted([(board[k][col], col) for col in range(n)], reverse=True)[:3]
                    
                    # Now, iterate over all possible combinations of one value from each row
                    # ensuring that the columns are unique
                    for val1, col1 in row1:
                        for val2, col2 in row2:
                            if col2 == col1:
                                continue
                            for val3, col3 in row3:
                                if col3 == col1 or col3 == col2:
                                    continue
                                current_sum = val1 + val2 + val3
                                if current_sum > max_sum:
                                    max_sum = current_sum
        return max_sum