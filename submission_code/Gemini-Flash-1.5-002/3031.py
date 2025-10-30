class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        n = len(grid)
        m = len(grid[0])
        result = [[0 for _ in range(m)] for _ in range(n)]
        
        for i in range(n):
            for j in range(m):
                product = 1
                for x in range(n):
                    for y in range(m):
                        if x != i or y != j:
                            product = (product * grid[x][y]) % 12345
                result[i][j] = product
        return result