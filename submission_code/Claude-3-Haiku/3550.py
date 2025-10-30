class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        max_sum = 0

        def is_valid(i, j):
            for x in range(m):
                if board[x][j] != -float('inf'):
                    return False
            for y in range(n):
                if board[i][y] != -float('inf'):
                    return False
            return True

        def backtrack(i, j, curr_sum, placed):
            nonlocal max_sum
            if placed == 3:
                max_sum = max(max_sum, curr_sum)
                return
            if i == m:
                return
            if j == n:
                backtrack(i + 1, 0, curr_sum, placed)
                return
            if board[i][j] == -float('inf'):
                backtrack(i, j + 1, curr_sum, placed)
                return
            if is_valid(i, j):
                backtrack(i, j + 1, curr_sum, placed)
                backtrack(i, j + 1, curr_sum + board[i][j], placed + 1)

        backtrack(0, 0, 0, 0)
        return max_sum