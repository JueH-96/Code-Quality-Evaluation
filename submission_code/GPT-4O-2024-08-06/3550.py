class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        from itertools import combinations
        
        m, n = len(board), len(board[0])
        max_sum = float('-inf')
        
        # Generate all possible combinations of 3 rows
        for rows in combinations(range(m), 3):
            # Generate all possible combinations of 3 columns
            for cols in combinations(range(n), 3):
                # Calculate the sum of the selected cells
                current_sum = sum(board[rows[i]][cols[i]] for i in range(3))
                max_sum = max(max_sum, current_sum)
        
        return max_sum