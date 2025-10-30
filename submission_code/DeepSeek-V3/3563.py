from itertools import permutations

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        # Generate all possible column selections for each row
        # Since we need to select one cell from each row, but no two in the same column
        # We need to find a permutation of columns where each row selects a unique column
        # However, since the number of rows and columns can be up to 10, the total permutations are 10! which is manageable
        
        # To handle cases where rows > cols, we need to select only 'cols' rows
        # But since the problem says to select one or more cells, and no two in the same row, we can select up to min(rows, cols) cells
        
        max_possible = min(rows, cols)
        
        # Generate all possible combinations of rows and columns
        # Since we need to select one cell from each row, but no two in the same column
        # We can think of it as selecting a permutation of columns for the selected rows
        
        # First, select which rows to use
        # Since we need to select up to max_possible rows, we can iterate over all possible combinations
        # For each combination, we need to find a permutation of columns that are unique
        
        max_score = 0
        
        # Iterate over the number of rows to select
        for r in range(1, max_possible + 1):
            # Generate all combinations of r rows
            from itertools import combinations
            row_combinations = combinations(range(rows), r)
            for rows_selected in row_combinations:
                # Now, for these rows, we need to select r columns, one for each row, with no duplicates
                # This is equivalent to finding a permutation of r columns
                # Since the number of columns is cols, and r <= cols, we can generate all permutations
                # of r columns from the cols available
                from itertools import permutations
                col_permutations = permutations(range(cols), r)
                for cols_selected in col_permutations:
                    # Check if all selected columns are unique
                    if len(set(cols_selected)) == r:
                        # Calculate the sum of the selected cells
                        current_sum = 0
                        used_values = set()
                        valid = True
                        for i in range(r):
                            row = rows_selected[i]
                            col = cols_selected[i]
                            value = grid[row][col]
                            if value in used_values:
                                valid = False
                                break
                            used_values.add(value)
                            current_sum += value
                        if valid and current_sum > max_score:
                            max_score = current_sum
        
        return max_score