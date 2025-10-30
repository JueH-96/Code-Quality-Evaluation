from typing import List
import itertools

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0]) if m > 0 else 0
        max_sum = float('-inf')
        
        # Precompute top cells for each row
        top_cells_per_row = []
        for row in board:
            # Sort cells in descending order of their values
            sorted_cells = sorted([(val, col) for col, val in enumerate(row)], reverse=True, key=lambda x: x[0])
            # Take top min(n, 20) cells
            top_cells_per_row.append(sorted_cells[:min(n, 20)])
        
        # Iterate through all combinations of three rows
        for row_indices in itertools.combinations(range(m), 3):
            rows = [top_cells_per_row[i] for i in row_indices]
            # Iterate through all combinations of one cell from each row
            for cells in itertools.product(*rows):
                cols = [cell[1] for cell in cells]
                if len(set(cols)) == 3:
                    current_sum = sum(cell[0] for cell in cells)
                    if current_sum > max_sum:
                        max_sum = current_sum
        return max_sum