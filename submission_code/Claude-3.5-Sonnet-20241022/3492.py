class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        if grid[0][0] == '.':
            return 0
            
        m, n = len(grid), len(grid[0])
        result = 0
        
        # For each possible bottom-right corner
        for i in range(m):
            for j in range(n):
                # For each possible top-left corner
                for r in range(i+1):
                    for c in range(j+1):
                        # Skip if top-left corner is not (0,0)
                        if r != 0 or c != 0:
                            continue
                            
                        # Count X's and Y's in current submatrix
                        x_count = 0
                        y_count = 0
                        
                        for p in range(r, i+1):
                            for q in range(c, j+1):
                                if grid[p][q] == 'X':
                                    x_count += 1
                                elif grid[p][q] == 'Y':
                                    y_count += 1
                                    
                        # Check conditions
                        if x_count > 0 and x_count == y_count:
                            result += 1
                            
        return result