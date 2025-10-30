class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        row_top = []
        # Precompute top 3 elements for each row
        for row in board:
            # Sort by value descending, then column ascending
            sorted_row = sorted([(val, idx) for idx, val in enumerate(row)], key=lambda x: (-x[0], x[1]))
            row_top.append(sorted_row[:3])
        
        max_sum = -float('inf')
        # Iterate all combinations of three distinct rows
        for i in range(m):
            for j in range(i + 1, m):
                for k in range(j + 1, m):
                    # Current rows are i, j, k
                    top_i = row_top[i]
                    top_j = row_top[j]
                    top_k = row_top[k]
                    # Check all combinations of elements from each row's top 3
                    for a in range(3):
                        for b in range(3):
                            for c in range(3):
                                # Get column indices
                                col_i = top_i[a][1]
                                col_j = top_j[b][1]
                                col_k = top_k[c][1]
                                # Check if all columns are distinct
                                if col_i != col_j and col_i != col_k and col_j != col_k:
                                    current_sum = top_i[a][0] + top_j[b][0] + top_k[c][0]
                                    if current_sum > max_sum:
                                        max_sum = current_sum
        return max_sum