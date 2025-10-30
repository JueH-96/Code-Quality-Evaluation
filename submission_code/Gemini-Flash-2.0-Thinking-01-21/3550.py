class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        rows = len(board)
        cols = len(board[0])
        max_sum = -float('inf')

        def solve(row_index, count, used_cols, current_sum):
            nonlocal max_sum
            if count == 3:
                max_sum = max(max_sum, current_sum)
                return

            if row_index == rows:
                return

            # Skip current row
            solve(row_index + 1, count, used_cols, current_sum)

            for col_index in range(cols):
                if not used_cols[col_index]:
                    used_cols[col_index] = True
                    solve(row_index + 1, count + 1, used_cols, current_sum + board[row_index][col_index])
                    used_cols[col_index] = False # Backtrack

        used_cols = [False] * cols
        solve(0, 0, used_cols, 0)

        return max_sum