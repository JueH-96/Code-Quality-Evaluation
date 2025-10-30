import math

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])

        overall_max_sum = -float('inf')

        # Precompute top K values and their column indices for each row.
        # K is chosen to be 3 for efficiency, as it typically covers
        # enough candidates without leading to TLE.
        # It's min(n, 3) to handle cases where n < 3 (though constraints say n >= 3,
        # it's good practice for other problems or similar contexts).
        K = 3 
        
        # row_sorted_vals[r] will store a list of (value, col_idx) tuples
        # for the top K values in row r, sorted descending by value.
        row_sorted_vals = [[] for _ in range(m)]

        for r in range(m):
            row_vals_with_idx = [(board[r][c], c) for c in range(n)]
            row_vals_with_idx.sort(key=lambda x: x[0], reverse=True)
            row_sorted_vals[r] = row_vals_with_idx[:K]

        # Iterate over all possible combinations of three distinct rows (r1, r2, r3)
        # C(m, 3) combinations of rows
        for r1 in range(m):
            for r2 in range(r1 + 1, m):
                for r3 in range(r2 + 1, m):
                    # We have chosen three distinct rows: r1, r2, r3
                    # Now we need to select three distinct columns c1, c2, c3
                    # and assign them to (r1, r2, r3) to maximize sum.
                    # This is a 3x3 assignment problem.

                    # Initialize max sum for this specific triplet of rows
                    current_max_for_rows = -float('inf')
                    
                    # Iterate through candidates for the first rook in r1
                    # (val1_r1 is board[r1][c1_val])
                    for val1_r1, c1_val in row_sorted_vals[r1]:
                        # Iterate through candidates for the second rook in r2
                        # (val2_r2 is board[r2][c2_val])
                        for val2_r2, c2_val in row_sorted_vals[r2]:
                            # Ensure c2_val is distinct from c1_val
                            if c2_val == c1_val: 
                                continue
                            
                            # Iterate through candidates for the third rook in r3
                            # (val3_r3 is board[r3][c3_val])
                            for val3_r3, c3_val in row_sorted_vals[r3]:
                                # Ensure c3_val is distinct from c1_val and c2_val
                                if c3_val == c1_val or c3_val == c2_val:
                                    continue
                                
                                # All three column indices are distinct. Calculate sum.
                                current_sum = val1_r1 + val2_r2 + val3_r3
                                current_max_for_rows = max(current_max_for_rows, current_sum)
                    
                    # Update the overall maximum sum
                    overall_max_sum = max(overall_max_sum, current_max_for_rows)
        
        return overall_max_sum