import itertools

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        if m < 3:
            return 0  # As per constraints, m >= 3, but handle just in case
        n = len(board[0])
        
        # Preprocess each row to get the top 3 cells (value, column index) sorted in descending order
        row_tops = []
        for row in board:
            cells = [(val, idx) for idx, val in enumerate(row)]
            cells.sort(reverse=True, key=lambda x: x[0])
            top3 = cells[:3]
            row_tops.append(top3)
        
        max_sum = -float('inf')
        
        # Iterate over all possible triplets of distinct rows
        for triplet in itertools.combinations(range(m), 3):
            r1, r2, r3 = triplet
            top_r1 = row_tops[r1]
            top_r2 = row_tops[r2]
            top_r3 = row_tops[r3]
            
            # Iterate through all possible cells in the top 3 of each row
            for c1 in top_r1:
                val1, col1 = c1
                for c2 in top_r2:
                    val2, col2 = c2
                    if col1 == col2:
                        continue  # Skip if columns are the same
                    for c3 in top_r3:
                        val3, col3 = c3
                        if col3 == col1 or col3 == col2:
                            continue  # Skip if columns are the same as previous
                        current_sum = val1 + val2 + val3
                        if current_sum > max_sum:
                            max_sum = current_sum
        
        return max_sum if max_sum != -float('inf') else 0