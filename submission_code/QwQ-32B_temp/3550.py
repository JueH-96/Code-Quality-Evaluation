class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        
        # Precompute top 3 values and their columns for each row
        top3 = []
        for r in range(m):
            row = board[r]
            # Create a list of tuples (value, column) for the row
            vals = [(row[c], c) for c in range(n)]
            # Sort in descending order of value
            vals.sort(reverse=True, key=lambda x: x[0])
            # Take top 3 elements
            top3_row = vals[:3]
            top3.append(top3_row)
        
        max_sum = -float('inf')
        
        # Iterate over all triplets of rows (r1, r2, r3) where r1 < r2 < r3
        for r1 in range(m):
            for r2 in range(r1 + 1, m):
                for r3 in range(r2 + 1, m):
                    current_max = -float('inf')
                    # Iterate over all combinations of top3 elements from each row
                    for a_val, a_col in top3[r1]:
                        for b_val, b_col in top3[r2]:
                            if a_col == b_col:
                                continue  # columns must be different
                            for c_val, c_col in top3[r3]:
                                if c_col == a_col or c_col == b_col:
                                    continue  # columns must be different
                                total = a_val + b_val + c_val
                                if total > current_max:
                                    current_max = total
                    if current_max > max_sum:
                        max_sum = current_max
        
        return max_sum