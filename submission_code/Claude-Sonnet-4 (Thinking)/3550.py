class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        from itertools import combinations
        
        m, n = len(board), len(board[0])
        
        max_sum = float('-inf')
        
        # Try all combinations of 3 rows and 3 columns
        for rows in combinations(range(m), 3):
            r0, r1, r2 = rows
            for cols in combinations(range(n), 3):
                c0, c1, c2 = cols
                # Try all 6 permutations of assigning rooks to positions
                current_max = max(
                    board[r0][c0] + board[r1][c1] + board[r2][c2],
                    board[r0][c0] + board[r1][c2] + board[r2][c1],
                    board[r0][c1] + board[r1][c0] + board[r2][c2],
                    board[r0][c1] + board[r1][c2] + board[r2][c0],
                    board[r0][c2] + board[r1][c0] + board[r2][c1],
                    board[r0][c2] + board[r1][c1] + board[r2][c0]
                )
                max_sum = max(max_sum, current_max)
        
        return max_sum