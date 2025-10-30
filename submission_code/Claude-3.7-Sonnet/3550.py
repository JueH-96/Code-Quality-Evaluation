from typing import List
from itertools import combinations, permutations

class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        
        max_sum = float('-inf')
        
        for rows in combinations(range(m), 3):
            for cols in combinations(range(n), 3):
                # Try all possible permutations of the columns
                curr_max = float('-inf')
                for perm in permutations(cols):
                    curr_sum = sum(board[rows[i]][perm[i]] for i in range(3))
                    curr_max = max(curr_max, curr_sum)
                
                max_sum = max(max_sum, curr_max)
        
        return max_sum