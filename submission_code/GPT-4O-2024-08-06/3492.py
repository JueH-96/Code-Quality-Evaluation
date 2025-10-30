class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        def count_equal_XY_subarrays(arr):
            count = 0
            balance = 0
            balance_map = {0: 1}
            
            for char in arr:
                if char == 'X':
                    balance += 1
                elif char == 'Y':
                    balance -= 1
                
                if balance in balance_map:
                    count += balance_map[balance]
                
                if balance in balance_map:
                    balance_map[balance] += 1
                else:
                    balance_map[balance] = 1
            
            return count
        
        rows, cols = len(grid), len(grid[0])
        total_count = 0
        
        for start_row in range(rows):
            col_balance = [0] * cols
            
            for end_row in range(start_row, rows):
                for col in range(cols):
                    if grid[end_row][col] == 'X':
                        col_balance[col] += 1
                    elif grid[end_row][col] == 'Y':
                        col_balance[col] -= 1
                
                total_count += count_equal_XY_subarrays(col_balance)
        
        return total_count