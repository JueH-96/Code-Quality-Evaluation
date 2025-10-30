class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        rows = len(board)
        cols = len(board[0])
        
        row_sums = [sum(row) for row in board]
        col_sums = [sum(col) for col in zip(*board)]
        
        row_max = max(row_sums)
        col_max = max(col_sums)
        
        row_sums.remove(row_max)
        col_sums.remove(col_max)
        
        second_row_max = max(row_sums)
        second_col_max = max(col_sums)
        
        return row_max + col_max + second_row_max + second_col_max