import sys
from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m_orig = len(board)
        n_orig = len(board[0])

        # Add transpose optimization if m_orig > n_orig
        transposed = False
        if m_orig > n_orig:
            # Transpose the board
            board = [list(row) for row in zip(*board)]
            m = n_orig  # new m
            n = m_orig  # new n
            transposed = True
        else:
            m = m_orig
            n = n_orig

        # Precompute sorted values and original indices for each row
        # row_sorted[i] is a list of (value, col_index) for row i, sorted descending by value
        row_sorted = []
        for i in range(m):
            row_data = [(board[i][j], j) for j in range(n)]
            row_data.sort(key=lambda x: x[0], reverse=True)
            row_sorted.append(row_data)

        max_sum = -sys.maxsize # Use a very small number for initialization

        # Iterate through all pairs of distinct rows {r1, r2}
        # There are C(m, 2) such pairs
        for r1 in range(m):
            # Iterate over rows after r1 to get distinct pairs {r1, r2}
            for r2 in range(r1 + 1, m):

                # Iterate through all pairs of distinct columns {c1, c2}
                # There are C(n, 2) such pairs
                for c1 in range(n):
                    # Iterate over columns after c1 to get distinct pairs {c1, c2}
                    for c2 in range(c1 + 1, n):

                        # Find the maximum value board[r3][c3] for r3 not in {r1, r2}, c3 not in {c1, c2}
                        # This value will be the third rook's contribution.
                        # The third rook must be in a row different from r1 and r2,
                        # and in a column different from c1 and c2.
                        max_third_val = -sys.maxsize # Sentinel value

                        # Iterate through all possible rows for the third rook
                        # Rows must be different from r1 and r2
                        for r3 in range(m):
                            # Skip rows already used by the first two potential rooks
                            if r3 == r1 or r3 == r2:
                                continue

                            # Find the maximum value in row r3 excluding columns c1 and c2
                            current_row_max = -sys.maxsize # Sentinel value

                            # Use the precomputed sorted list for row r3
                            # Iterate through the sorted values until we find a column not in {c1, c2}
                            # Since n >= 3 (constraint), there will always be at least one column != c1 and != c2.
                            # Thus, we are guaranteed to find a valid col_idx unless the row is completely unusable
                            # (e.g., all values are -sys.maxsize, which is fine) or the row is empty (not possible).
                            for val, col_idx in row_sorted[r3]:
                                # Check if the current column index is one of the excluded columns
                                if col_idx != c1 and col_idx != c2:
                                    # Found the best value in row r3 that is in an allowed column
                                    current_row_max = val
                                    break # We only need the single maximum from this row

                            # If a valid column was found in this row r3 (i.e., current_row_max was updated)
                            if current_row_max != -sys.maxsize:
                                # Update the maximum value found so far for the third rook position
                                max_third_val = max(max_third_val, current_row_max)

                        # If a valid third rook position was found anywhere outside rows {r1, r2}
                        # and columns {c1, c2} (guaranteed if m >= 3 and n >= 3),
                        # max_third_val will be greater than the sentinel value.
                        if max_third_val != -sys.maxsize:
                            # We have fixed two distinct rows {r1, r2} and two distinct columns {c1, c2},
                            # and found the maximum possible value `max_third_val` for a third rook
                            # placed in a row not in {r1, r2} and a column not in {c1, c2}.
                            # Let the cell giving `max_third_val` be (r3_best, c3_best).
                            # The three rooks will occupy rows {r1, r2, r3_best} and columns {c_a, c_b, c3_best}.
                            # The rows are distinct by construction ({r1, r2} distinct, r3_best != r1, r2).
                            # The columns must be distinct. {c_a, c_b} must be {c1, c2}, and c3_best must be distinct from c1, c2.
                            # This is satisfied by how max_third_val was found (from columns not in {c1, c2}).

                            # We need to consider how the first two rooks are placed in rows {r1, r2} using columns from {c1, c2}.
                            # There are two non-attacking placements:
                            # 1. Rook 1 at (r1, c1), Rook 2 at (r2, c2). The third rook is at (r3_best, c3_best).
                            current_sum1 = board[r1][c1] + board[r2][c2] + max_third_val
                            max_sum = max(max_sum, current_sum1)

                            # 2. Rook 1 at (r1, c2), Rook 2 at (r2, c1). The third rook is at (r3_best, c3_best).
                            current_sum2 = board[r1][c2] + board[r2][c1] + max_third_val
                            max_sum = max(max_sum, current_sum2)

        # Since constraints m, n >= 3, at least one valid configuration of 3 non-attacking rooks exists.
        # Thus, max_sum will be updated at least once from its initial sentinel value.

        return max_sum