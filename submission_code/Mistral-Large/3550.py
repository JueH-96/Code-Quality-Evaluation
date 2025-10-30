from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])

        # Sort rows based on the maximum value in each row
        board.sort(key=lambda row: max(row), reverse=True)

        # Initialize a list to keep track of the maximum sum achievable
        max_sum = 0
        used_columns = set()

        for row in board:
            max_value = -float('inf')
            max_col = -1
            for col in range(n):
                if col not in used_columns and row[col] > max_value:
                    max_value = row[col]
                    max_col = col
            if max_col != -1:
                max_sum += max_value
                used_columns.add(max_col)

        return max_sum

# Example usage:
# board = [[-3,1,1,1],[-3,1,-3,1],[-3,2,1,1]]
# solution = Solution()
# print(solution.maximumValueSum(board))  # Output: 4