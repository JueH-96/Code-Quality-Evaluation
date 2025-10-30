class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if n == 0:
            return 0
        m = len(grid[0])
        
        # Pre-sort each row's values in descending order
        sorted_rows = []
        for row in grid:
            sorted_row = sorted(row, reverse=True)
            sorted_rows.append(sorted_row)
        
        max_score = 0
        
        def backtrack(row, used, current_sum):
            nonlocal max_score
            # Calculate the upper bound for this path
            upper = current_sum
            for i in range(row, n):
                for val in sorted_rows[i]:
                    if val not in used:
                        upper += val
                        break
            if upper <= max_score:
                return
            
            if row == n:
                if current_sum > max_score:
                    max_score = current_sum
                return
            
            # Option 1: Skip this row
            backtrack(row + 1, used, current_sum)
            
            # Option 2: Choose a cell from this row (try in descending order)
            for val in sorted_rows[row]:
                if val not in used:
                    new_used = used.union({val})
                    backtrack(row + 1, new_used, current_sum + val)
                    # Early exit if the current path's upper bound is already the max_score
                    # This is an optimization but may not always be safe, so commented out for correctness
                    # if current_sum + val + (upper - current_sum - val) == max_score:
                    #     break
        
        backtrack(0, frozenset(), 0)
        return max_score