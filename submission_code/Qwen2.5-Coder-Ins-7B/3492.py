class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])
        count = 0
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 'X':
                    count += 1
                    for k in range(i + 1, rows):
                        if grid[k][j] == 'X':
                            count += 1
                        else:
                            break
                    for l in range(j + 1, cols):
                        if grid[i][l] == 'X':
                            count += 1
                        else:
                            break
                    for k in range(i + 1, rows):
                        for l in range(j + 1, cols):
                            if grid[k][l] == 'X':
                                count += 1
                            else:
                                break
        return count