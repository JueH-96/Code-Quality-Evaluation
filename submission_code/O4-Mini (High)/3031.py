from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        M = 12345
        n = len(grid)
        if n == 0:
            return []
        m = len(grid[0])
        N = n * m

        # Flatten grid values modulo M
        arr = [0] * N
        idx = 0
        for row in grid:
            for val in row:
                arr[idx] = val % M
                idx += 1

        # Build prefix products: prefix[i] = product of arr[0..i-1] mod M
        prefix = [1] * (N + 1)
        for i in range(N):
            prefix[i + 1] = (prefix[i] * arr[i]) % M

        # Build result by carrying a running suffix product
        res = [[0] * m for _ in range(n)]
        suffix = 1  # suffix = product of arr[k+1..N-1] mod M at each step
        for k in range(N - 1, -1, -1):
            i = k // m
            j = k % m
            # product of everything except arr[k] is prefix[k] * suffix
            res[i][j] = (prefix[k] * suffix) % M
            # incorporate arr[k] into suffix for the next iteration
            suffix = (suffix * arr[k]) % M

        return res