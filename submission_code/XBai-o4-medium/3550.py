import itertools
from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0])
        
        # Preprocess each row to get top 3 (value, column)
        row_top3 = []
        for i in range(m):
            entries = [ (board[i][j], j) for j in range(n) ]
            # Sort in descending order of value, then column ascending
            sorted_entries = sorted(entries, key=lambda x: (-x[0], x[1]))
            row_top3.append(sorted_entries[:3])
        
        max_total = -float('inf')
        
        # Iterate through all possible triplets of rows
        for rows in itertools.combinations(range(m), 3):
            i, j, k = rows
            # Get top3 lists for each row
            list_i = row_top3[i]
            list_j = row_top3[j]
            list_k = row_top3[k]
            # Check all combinations of one entry from each list
            for v1, c1 in list_i:
                for v2, c2 in list_j:
                    for v3, c3 in list_k:
                        if c1 != c2 and c2 != c3 and c1 != c3:
                            current_sum = v1 + v2 + v3
                            if current_sum > max_total:
                                max_total = current_sum
        
        return max_total