from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        n = len(grid)
        # Assuming rectangular grid as per problem statement
        m = len(grid[0])  
        L = n * m

        # Flatten the grid into a 1D array
        arr = [0] * L
        idx = 0
        for i in range(n):
            for j in range(m):
                arr[idx] = grid[i][j]
                idx += 1

        # Compute prefix products: pre[i] = product of arr[0..i-1] mod MOD
        pre = [1] * (L + 1)
        for i in range(L):
            pre[i+1] = (pre[i] * arr[i]) % MOD

        # Compute suffix products: suf[i] = product of arr[i..L-1] mod MOD
        suf = [1] * (L + 1)
        for i in range(L - 1, -1, -1):
            suf[i] = (suf[i+1] * arr[i]) % MOD

        # Build the answer by excluding each element via prefix*suffix
        ans = [[0] * m for _ in range(n)]
        for k in range(L):
            i, j = divmod(k, m)
            ans[i][j] = (pre[k] * suf[k+1]) % MOD

        return ans