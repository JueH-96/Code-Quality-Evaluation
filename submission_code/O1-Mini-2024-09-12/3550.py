from typing import List
from itertools import combinations

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        # Precompute the top 3 cells for each row
        top_cells_per_row = []
        for row in board:
            # Pair each cell value with its column index
            paired = list(zip(row, range(len(row))))
            # Sort the paired list in descending order based on cell values
            sorted_paired = sorted(paired, key=lambda x: x[0], reverse=True)
            # Append the top 3 cells (value and column index) for the current row
            top_cells_per_row.append(sorted_paired[:3])
        
        max_sum = -float('inf')
        m = len(board)
        
        # Iterate over all combinations of 3 distinct rows
        for rows in combinations(range(m), 3):
            # Retrieve the top cells for each of the three selected rows
            cells1 = top_cells_per_row[rows[0]]
            cells2 = top_cells_per_row[rows[1]]
            cells3 = top_cells_per_row[rows[2]]
            
            # Iterate through all possible selections of one cell from each row
            for val1, col1 in cells1:
                for val2, col2 in cells2:
                    if col2 == col1:
                        continue  # Columns must be distinct
                    for val3, col3 in cells3:
                        if col3 == col1 or col3 == col2:
                            continue  # Columns must be distinct
                        current_sum = val1 + val2 + val3
                        if current_sum > max_sum:
                            max_sum = current_sum
        
        return max_sum