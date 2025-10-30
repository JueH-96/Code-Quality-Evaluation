class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        import math
        
        m = len(board)
        n = len(board[0])
        
        # For each row, determine the indices of up to the top 5 columns
        # by cell value in descending order. We only need top 5 because
        # at most two columns could be "blocked" when placing 3 rooks.
        # (If n < 5, just take all columns.)
        top_cols_per_row = []
        k = min(n, 5)
        for r in range(m):
            # Sort columns by descending board[r][c], then keep top k
            cols_sorted = sorted(range(n), key=lambda c: board[r][c], reverse=True)
            top_cols_per_row.append(cols_sorted[:k])
        
        max_sum = float('-inf')
        
        # Try all combinations of three distinct rows r1 < r2 < r3
        for r1 in range(m - 2):
            for r2 in range(r1 + 1, m - 1):
                for r3 in range(r2 + 1, m):
                    # For each of those row triples, attempt placing rooks
                    # in distinct columns (c1, c2, c3) chosen from each row's top columns.
                    for c1 in top_cols_per_row[r1]:
                        for c2 in top_cols_per_row[r2]:
                            if c2 == c1:
                                continue
                            for c3 in top_cols_per_row[r3]:
                                if c3 == c1 or c3 == c2:
                                    continue
                                s = board[r1][c1] + board[r2][c2] + board[r3][c3]
                                if s > max_sum:
                                    max_sum = s
        
        return max_sum