from itertools import combinations
from typing import List

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m = len(board)
        n = len(board[0]) if m else 0
        preprocessed = []
        
        for row in board:
            cols = [(val, idx) for idx, val in enumerate(row)]
            cols.sort(key=lambda x: (-x[0], x[1]))  # Sort by descending value, then ascending column
            preprocessed.append(cols[:3])  # Keep top 3 entries
        
        max_sum = -float('inf')
        
        # Iterate through all combinations of 3 distinct rows
        for rows in combinations(range(m), 3):
            i, j, k = rows
            a_list = preprocessed[i]
            b_list = preprocessed[j]
            c_list = preprocessed[k]
            
            for a_val, a_col in a_list:
                for b_val, b_col in b_list:
                    for c_val, c_col in c_list:
                        if a_col != b_col and b_col != c_col and a_col != c_col:
                            current_sum = a_val + b_val + c_val
                            if current_sum > max_sum:
                                max_sum = current_sum
        
        return max_sum