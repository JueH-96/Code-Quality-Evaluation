class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        if n == 0:
            return []
        m = len(grid[0])
        result = []
        for i in range(n):
            row = []
            for j in range(m):
                product = 1
                for x in range(n):
                    for y in range(m):
                        if x == i and y == j:
                            continue
                        product = (product * grid[x][y]) % 12345
                row.append(product % 12345)
            result.append(row)
        return result