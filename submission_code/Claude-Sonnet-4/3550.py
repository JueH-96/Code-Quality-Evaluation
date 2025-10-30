class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        from itertools import combinations, permutations
        
        m, n = len(board), len(board[0])
        max_sum = float('-inf')
        
        # Try all combinations of 3 rows
        for rows in combinations(range(m), 3):
            # Try all combinations of 3 columns
            for cols in combinations(range(n), 3):
                # Try all permutations of assigning rows to columns
                for col_perm in permutations(cols):
                    current_sum = 0
                    for i in range(3):
                        current_sum += board[rows[i]][col_perm[i]]
                    max_sum = max(max_sum, current_sum)
        
        return max_sum