# from typing import List # For older Python versions

class Solution:
  def maxScore(self, grid: list[list[int]]) -> int:
    R = len(grid)
    C = len(grid[0])

    # Preprocessing step 1: Find all distinct values in the grid and map them to indices 0..D-1.
    # This allows using a bitmask for the set of used values.
    # D is the number of distinct values.
    # Using a generator expression for conciseness in flattening the grid.
    distinct_values_in_grid = sorted(list(set(val for row_vals in grid for val in row_vals)))
    
    value_to_idx_map = {val: i for i, val in enumerate(distinct_values_in_grid)}
    
    # Memoization table: stores results for states (current_row_index, used_values_mask)
    memo = {}

    # Preprocessing step 2 (optional optimization): For each row, precompute unique (value, mapped_index) pairs.
    # This avoids redundant checks if a value appears multiple times in the same row
    # and saves recalculating mapped_index repeatedly.
    # Each element in processed_grid_rows will be a list of tuples: [(value, mapped_idx), ...]
    processed_grid_rows = []
    for r in range(R):
        unique_cells_in_row = []
        # Iterate over unique values in grid[r]
        for val_in_cell in sorted(list(set(grid[r]))): # Sort for determinism, though not strictly needed for max score
            unique_cells_in_row.append((val_in_cell, value_to_idx_map[val_in_cell]))
        processed_grid_rows.append(unique_cells_in_row)

    # Recursive function to find the maximum score
    def find_max_score_recursive(current_row_idx: int, used_values_mask: int) -> int:
      # Base case: If all rows have been processed, no more score can be added.
      if current_row_idx == R:
        return 0

      # Check if this state has already been computed.
      state = (current_row_idx, used_values_mask)
      if state in memo:
        return memo[state]

      # Option 1: Skip the current row.
      # The score is whatever we can get from the remaining rows by not picking from current_row_idx.
      max_achievable_score = find_max_score_recursive(current_row_idx + 1, used_values_mask)

      # Option 2: Select a cell from the current row.
      # Iterate through the preprocessed unique (value, mapped_idx) pairs for the current row.
      for value, mapped_idx in processed_grid_rows[current_row_idx]:
        # Check if this value (via its mapped_idx) is already in used_values_mask.
        if not (used_values_mask & (1 << mapped_idx)):  # Test if bit for mapped_idx is 0
          # If the value is not used, we can select it.
          # Add its contribution and recurse for the next row with an updated mask.
          # (used_values_mask | (1 << mapped_idx)) sets the bit for mapped_idx to 1.
          score_from_this_choice = value + find_max_score_recursive(current_row_idx + 1, used_values_mask | (1 << mapped_idx))
          
          # Update max_achievable_score if this choice leads to a better score.
          if score_from_this_choice > max_achievable_score:
            max_achievable_score = score_from_this_choice
      
      # Store the result for this state in the memoization table and return it.
      memo[state] = max_achievable_score
      return max_achievable_score

    # Initial call: start from row 0 with no values used yet (mask is 0).
    return find_max_score_recursive(0, 0)