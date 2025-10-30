class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        if rows == 0:
            return 0
        
        # Preprocess each row to get unique values sorted in descending order
        row_values = []
        for r in grid:
            unique_vals = sorted(list(set(r)), reverse=True)
            row_values.append(unique_vals)
        
        # Compute max_val for each row (highest possible value in the row)
        max_val = []
        for rv in row_values:
            if rv:
                max_val.append(rv[0])
            else:
                max_val.append(0)  # based on constraints, this case won't occur
        
        # Compute sum_max: sum_max[i] is the maximum possible sum from row i to the end
        sum_max = [0] * (rows + 1)
        for i in range(rows - 1, -1, -1):
            sum_max[i] = max_val[i] + sum_max[i + 1]
        
        max_sum = 0
        
        def backtrack(row, used, current_sum):
            nonlocal max_sum
            if row == rows:
                if current_sum > max_sum:
                    max_sum = current_sum
                return
            # Prune if current_sum + sum_max[row] can't exceed current max_sum
            if current_sum + sum_max[row] <= max_sum:
                return
            # Try each possible value in this row
            for val in row_values[row]:
                if val not in used:
                    used.add(val)
                    backtrack(row + 1, used, current_sum + val)
                    used.remove(val)
        
        backtrack(0, set(), 0)
        return max_sum