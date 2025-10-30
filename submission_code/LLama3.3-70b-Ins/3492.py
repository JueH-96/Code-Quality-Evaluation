from typing import List

class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        count = 0
        
        # Iterate over all possible submatrices
        for i in range(m):
            for j in range(n):
                for x in range(i, m):
                    for y in range(j, n):
                        submatrix = [row[j:y+1] for row in grid[i:x+1]]
                        
                        # Count 'X' and 'Y' in the submatrix
                        x_count = sum(row.count('X') for row in submatrix)
                        y_count = sum(row.count('Y') for row in submatrix)
                        
                        # Check if the submatrix has an equal frequency of 'X' and 'Y' and at least one 'X'
                        if x_count == y_count and x_count > 0:
                            count += 1
        
        return count