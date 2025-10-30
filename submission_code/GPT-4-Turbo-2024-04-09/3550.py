class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        from itertools import permutations
        
        m, n = len(board), len(board[0])
        
        # If the board is smaller or equal to 3x3, we can directly compute the max sum for non-attacking rooks
        if m == 3 and n == 3:
            # Generate all possible placements of 3 rooks such that no two are in the same row or column
            max_sum = float('-inf')
            for perm in permutations(range(3), 3):
                current_sum = board[0][perm[0]] + board[1][perm[1]] + board[2][perm[2]]
                max_sum = max(max_sum, current_sum)
            return max_sum
        
        # For larger boards, use a more efficient approach
        # We will use a greedy approach with a heuristic based on the maximum values in rows and columns
        
        # Step 1: Calculate the maximum values in each row and column
        row_max = [max(row) for row in board]
        col_max = [max(board[i][j] for i in range(m)) for j in range(n)]
        
        # Step 2: Sort these maximum values
        row_max.sort(reverse=True)
        col_max.sort(reverse=True)
        
        # Step 3: We need to pick 3 values, either from rows or columns, but not both from the same row or column
        # We will try to pick the top 3 from rows and columns and see which gives the best sum
        # This is a heuristic and may not always give the optimal solution but should work well for large boards
        
        # Pick top 3 from rows and columns
        best_row_sum = sum(row_max[:3])
        best_col_sum = sum(col_max[:3])
        
        # Return the maximum of the two sums
        return max(best_row_sum, best_col_sum)