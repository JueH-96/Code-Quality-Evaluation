class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        ans = 0
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 'X':
                    grid[i][j] = 1
                else:
                    grid[i][j] = 0
                if i > 0:
                    grid[i][j] += grid[i-1][j]
        
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    continue
                cnt = Counter()
                for k in range(j, n):
                    if grid[i][k] == 0:
                        break
                    cnt[grid[i][k]] += 1
                    if cnt[grid[i][k]] == cnt[grid[i][k]-1]:
                        ans += 1
        return ans