class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        n = len(grid)
        m = len(grid[0])
        total = [[0]*(m+1) for _ in range(n+1)]
        for i in range(1, n+1):
            for j in range(1, m+1):
                total[i][j] = total[i-1][j] + total[i][j-1] - total[i-1][j-1] + (grid[i-1][j-1] == 'B')

        def check(x1, y1, x2, y2):
            return total[x2][y2] - total[x1-1][y2] - total[x2][y1-1] + total[x1-1][y1-1]

        for i in range(2, min(n, m)+1):
            if n % i == 0 or m % i == 0:
                for x in range(i, n+1):
                    for y in range(i, m+1):
                        if check(x-i+1, y-i+1, x, y) * 4 == i * i:
                            return True
                        if check(x-i+1, y-i+1, x, y) * 2 == i * i and (n * m - check(1, 1, n, m)) * 2 == i * i:
                            return True
        return False