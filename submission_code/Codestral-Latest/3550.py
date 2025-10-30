class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])

        # Precompute the maximum value for each row and column
        max_row = [max(row) for row in board]
        max_col = [max(board[i][j] for i in range(m)) for j in range(n)]

        # Function to get the sum of values for three rooks placed in non-attacking positions
        def get_sum(r1, c1, r2, c2, r3, c3):
            return board[r1][c1] + board[r2][c2] + board[r3][c3]

        max_sum = float('-inf')

        # Iterate through all possible positions for the first rook
        for r1 in range(m):
            for c1 in range(n):
                # Iterate through all possible positions for the second rook
                for r2 in range(m):
                    for c2 in range(n):
                        if r2 != r1 and c2 != c1:
                            # Iterate through all possible positions for the third rook
                            for r3 in range(m):
                                for c3 in range(n):
                                    if r3 != r1 and r3 != r2 and c3 != c1 and c3 != c2:
                                        max_sum = max(max_sum, get_sum(r1, c1, r2, c2, r3, c3))

        return max_sum