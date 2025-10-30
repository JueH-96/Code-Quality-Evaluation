class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        
        def backtrack(row, used_cols, count, curr_sum):
            if count == 3:
                return curr_sum
                
            if row >= m:
                return float('-inf')
                
            max_sum = float('-inf')
            
            # Try placing rook in each column of current row
            for col in range(n):
                if col not in used_cols:
                    used_cols.add(col)
                    # Try placing rook at current position
                    max_sum = max(max_sum, backtrack(row + 1, used_cols, count + 1, curr_sum + board[row][col]))
                    # Skip current row and try next row
                    max_sum = max(max_sum, backtrack(row + 1, used_cols, count, curr_sum))
                    used_cols.remove(col)
            
            # Skip current row entirely
            max_sum = max(max_sum, backtrack(row + 1, used_cols, count, curr_sum))
            
            return max_sum
            
        return backtrack(0, set(), 0, 0)