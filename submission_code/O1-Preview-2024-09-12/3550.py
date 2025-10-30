class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        K = 250  # Adjust K based on acceptable time and memory
        
        # Collect all cells with their values, rows, and columns
        cells = []
        for i in range(m):
            for j in range(n):
                cells.append((-board[i][j], i, j))  # Negate value for descending sort
        # Sort cells based on values descending
        cells.sort()
        cells = [(-val, row, col) for val, row, col in cells]  # Restore original values
        
        # Keep top K cells
        cells = cells[:K]
        
        max_sum = None
        K = len(cells)
        for i in range(K):
            val1, row1, col1 = cells[i]
            for j in range(i+1, K):
                val2, row2, col2 = cells[j]
                if row1 == row2 or col1 == col2:
                    continue
                prohibited_rows = {row1, row2}
                prohibited_cols = {col1, col2}
                for k in range(K):
                    if k == i or k == j:
                        continue
                    val3, row3, col3 = cells[k]
                    if row3 in prohibited_rows or col3 in prohibited_cols:
                        continue
                    total = val1 + val2 + val3
                    if max_sum is None or total > max_sum:
                        max_sum = total
                    break  # Found best c3 for this pair (c1, c2)
        # If max_sum is still None (edge case), pick any three non-conflicting cells
        if max_sum is None:
            # Try to pick any three non-conflicting cells
            for i in range(K):
                val1, row1, col1 = cells[i]
                for j in range(i+1, K):
                    val2, row2, col2 = cells[j]
                    if row1 == row2 or col1 == col2:
                        continue
                    for k in range(j+1, K):
                        val3, row3, col3 = cells[k]
                        if row3 in {row1, row2} or col3 in {col1, col2}:
                            continue
                        total = val1 + val2 + val3
                        if max_sum is None or total > max_sum:
                            max_sum = total
                        break
        # If still None (should not happen), pick any three distinct cells
        if max_sum is None:
            max_sum = sum(cells[i][0] for i in range(3))
        return max_sum