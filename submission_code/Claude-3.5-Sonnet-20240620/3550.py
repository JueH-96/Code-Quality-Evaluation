class Solution:
    def maximumValueSum(self, board: List[List[int]]) -> int:
        m, n = len(board), len(board[0])
        
        def dfs(row, col, used_cols):
            if row == m:
                return 0
            
            max_sum = float('-inf')
            for j in range(n):
                if j not in used_cols:
                    used_cols.add(j)
                    max_sum = max(max_sum, board[row][j] + dfs(row + 1, j, used_cols))
                    used_cols.remove(j)
            
            return max_sum
        
        return dfs(0, 0, set())