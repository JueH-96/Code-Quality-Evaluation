class Solution:
    def numberOfSubmatrices(self, grid: List[List[str]]) -> int:
        m, n = len(grid), len(grid[0])
        prefix = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if grid[i - 1][j - 1] == 'X':
                    prefix[i][j] = prefix[i][j - 1] + 1
        res = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cnt = float('inf')
                for k in range(i, 0, -1):
                    cnt = min(cnt, prefix[k][j])
                    col = j - prefix[i][j] + prefix[k - 1][j]
                    res += min(col, cnt)
        return res