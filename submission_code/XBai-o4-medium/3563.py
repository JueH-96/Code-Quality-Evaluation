from typing import List

class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        # Preprocess each row: sort in descending order, deduplicate
        sorted_rows = []
        max_values = []
        for row in grid:
            unique = list(set(row))
            sorted_row = sorted(unique, reverse=True)
            sorted_rows.append(sorted_row)
            max_values.append(sorted_row[0] if sorted_row else 0)  # sorted_row can't be empty
        
        n = len(grid)
        # Precompute suffix sums
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = max_values[i] + suffix_sums[i + 1]
        
        best = [0]
        
        def backtrack(row: int, selected: set, current_sum: int):
            if row == n:
                if current_sum > best[0]:
                    best[0] = current_sum
                return
            
            # Calculate upper bound
            remaining_sum = suffix_sums[row]
            upper_bound = current_sum + remaining_sum
            
            if upper_bound <= best[0]:
                return  # Prune
            
            # Option 1: skip this row
            backtrack(row + 1, selected, current_sum)
            
            # Option 2: take one of the cells in this row, if possible
            current_row_values = sorted_rows[row]
            for val in current_row_values:
                if val not in selected:
                    selected.add(val)
                    backtrack(row + 1, selected, current_sum + val)
                    selected.remove(val)
        
        backtrack(0, set(), 0)
        return best[0]