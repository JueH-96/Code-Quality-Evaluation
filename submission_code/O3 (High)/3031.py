from typing import List

class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        MOD = 12345
        
        # --------- flatten the grid and keep every value modulo MOD ----------
        flat = []
        for row in grid:
            for val in row:
                flat.append(val % MOD)
        
        L = len(flat)                           # total number of elements
        
        # -------------------- prefix products --------------------------------
        prefix = [1]*(L+1)                      # prefix[i] = product of flat[0..i-1]  (mod MOD)
        for i in range(L):
            prefix[i+1] = (prefix[i] * flat[i]) % MOD
        
        # -------------------- suffix products --------------------------------
        suffix = [1]*(L+1)                      # suffix[i] = product of flat[i..L-1] (mod MOD)
        for i in range(L-1, -1, -1):
            suffix[i] = (suffix[i+1] * flat[i]) % MOD
        
        # -------------------- result for every position ----------------------
        prod_except_self = [(prefix[i] * suffix[i+1]) % MOD for i in range(L)]
        
        # -------------------- rebuild 2-D matrix in original shape -----------
        ans, idx = [], 0
        for row in grid:
            cur = []
            for _ in row:
                cur.append(prod_except_self[idx])
                idx += 1
            ans.append(cur)
        
        return ans