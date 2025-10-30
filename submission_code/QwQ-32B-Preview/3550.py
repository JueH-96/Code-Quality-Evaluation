from typing import List
import heapq
import itertools

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        max_sum = float('-inf')
        
        # Precompute the indices of the top three columns for each row
        top3_cols = []
        for row in board:
            # Get the top 3 columns with the highest values
            indexed_values = [(val, j) for j, val in enumerate(row)]
            top3 = heapq.nlargest(3, indexed_values)
            top3_cols.append([col for val, col in top3])
        
        # Iterate over all combinations of three rows
        for rows in itertools.combinations(range(m), 3):
            row1, row2, row3 = rows
            # Get the top three columns for each of the selected rows
            cols1 = top3_cols[row1]
            cols2 = top3_cols[row2]
            cols3 = top3_cols[row3]
            # Iterate over all possible assignments of three distinct columns
            for col1 in cols1:
                for col2 in cols2:
                    if col2 == col1:
                        continue  # Skip if column is already used
                    for col3 in cols3:
                        if col3 == col1 or col3 == col2:
                            continue  # Skip if column is already used
                        # Calculate the sum of the selected cells
                        current_sum = board[row1][col1] + board[row2][col2] + board[row3][col3]
                        # Update the maximum sum if the current sum is greater
                        if current_sum > max_sum:
                            max_sum = current_sum
        return max_sum