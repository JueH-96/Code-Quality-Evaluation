from itertools import permutations

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        # Generate all possible column selections where each row selects one column
        # Since rows and cols are small (<=10), we can use permutations
        # We need to select one column for each row, ensuring no two rows select the same column
        # This is equivalent to finding a permutation of columns for the rows
        
        # Since the number of rows and columns can be different, we need to handle cases where rows > cols or cols > rows
        # For rows > cols, some rows will not select any column
        # For cols > rows, some columns will not be selected
        
        # To handle this, we can consider all possible ways to select min(rows, cols) columns and assign them to rows
        
        # Let's find the minimum of rows and cols
        min_rc = min(rows, cols)
        
        # Generate all possible combinations of columns
        # We need to select min_rc columns from cols columns
        from itertools import combinations
        
        max_score = 0
        
        # Iterate over all possible combinations of columns
        for cols_selected in combinations(range(cols), min_rc):
            # Now, assign these columns to rows
            # We need to assign each column to a unique row
            # This is equivalent to finding a permutation of rows for the selected columns
            # Since the number of selected columns is min_rc, and the number of rows is rows, we need to select min_rc rows
            # So, we need to select min_rc rows from rows rows
            # Then, assign the selected columns to these rows in all possible ways
            
            # Generate all possible row selections
            for rows_selected in combinations(range(rows), min_rc):
                # Now, assign the columns to these rows in all possible ways
                # This is equivalent to finding a permutation of the selected columns for the selected rows
                for perm in permutations(cols_selected):
                    # Now, assign the columns to the rows
                    # Create a set to keep track of the selected values
                    selected_values = set()
                    score = 0
                    valid = True
                    for i in range(min_rc):
                        row = rows_selected[i]
                        col = perm[i]
                        value = grid[row][col]
                        if value in selected_values:
                            valid = False
                            break
                        selected_values.add(value)
                        score += value
                    if valid and score > max_score:
                        max_score = score
        
        return max_score