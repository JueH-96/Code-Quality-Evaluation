class Solution:
    def maximumValueSum(self, board: list[list[int]]) -> int:
        m, n = len(board), len(board[0])
        row_max = [max(row) for row in board]
        col_max = [max(col) for col in zip(*board)]
        rook_values = []

        for _ in range(3):
            max_row_value = max(row_max)
            max_col_value = max(col_max)
            if max_row_value >= max_col_value:
                max_value = max_row_value
                max_index = row_max.index(max_value)
                row_max[max_index] = float('-inf')
                col_index = board[max_index].index(max_value)
                for i in range(m):
                    board[i][col_index] = float('-inf')
            else:
                max_value = max_col_value
                max_index = col_max.index(max_value)
                col_max[max_index] = float('-inf')
                row_index = [row[max_index] for row in board].index(max_value)
                board[row_index] = [float('-inf')] * n
            rook_values.append(max_value)

        return sum(rook_values)