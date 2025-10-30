class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        from itertools import combinations, permutations
        
        m = len(board)
        n = len(board[0])
        max_sum = float('-inf')
        
        # Choose 3 rows out of m rows
        for rows in combinations(range(m), 3):
            # Choose 3 columns out of n columns
            for cols in combinations(range(n), 3):
                # Try all permutations of columns for the selected rows
                for perm in permutations(cols):
                    current_sum = 0
                    for i in range(3):
                        current_sum += board[rows[i]][perm[i]]
                    max_sum = max(max_sum, current_sum)
        
        return max_sum