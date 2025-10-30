class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        
        # For each row, find the top 3 columns (largest values).
        # Since we place only one rook per row and we have 3 rooks total,
        # we only need each row's best 3 columns (the 3 highest values).
        rowTop3 = []
        for r in range(m):
            # Sort columns in this row by descending value
            row_vals = sorted(((board[r][c], c) for c in range(n)), key=lambda x: x[0], reverse=True)
            # Keep only up to the top 3
            rowTop3.append(row_vals[:3])
        
        max_sum = float('-inf')
        
        # We must pick 3 distinct rows to place the 3 rooks.
        # Enumerate all combinations of three different rows: O(m^3).
        for r1 in range(m - 2):
            for r2 in range(r1 + 1, m - 1):
                for r3 in range(r2 + 1, m):
                    # Now try all combinations of each row's top-3 columns.
                    # That is at most 3 x 3 x 3 = 27 checks for each triple (r1, r2, r3).
                    for val1, c1 in rowTop3[r1]:
                        for val2, c2 in rowTop3[r2]:
                            # Columns must be distinct
                            if c2 == c1:
                                continue
                            for val3, c3 in rowTop3[r3]:
                                if c3 == c1 or c3 == c2:
                                    continue
                                current_sum = val1 + val2 + val3
                                if current_sum > max_sum:
                                    max_sum = current_sum
        
        return max_sum