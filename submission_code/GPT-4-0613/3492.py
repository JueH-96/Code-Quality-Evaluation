class Solution:
    def numberOfSubmatrices(self, grid):
        m, n = len(grid), len(grid[0])
        count = 0
        for i in range(m):
            for j in range(n):
                for x in range(i, m):
                    for y in range(j, n):
                        submatrix = [row[j:y+1] for row in grid[i:x+1]]
                        flat_submatrix = [item for sublist in submatrix for item in sublist]
                        if flat_submatrix.count('X') == flat_submatrix.count('Y') and 'X' in flat_submatrix:
                            count += 1
        return count