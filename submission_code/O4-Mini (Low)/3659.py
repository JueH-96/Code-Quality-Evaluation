from typing import List

class Solution:
    def countPathsWithXorValue(self, grid: List[List[int]], k: int) -> int:
        MOD = 10**9 + 7
        m, n = len(grid), len(grid[0])
        total = m + n - 2
        mid = total // 2
        
        # dp1[i][j][x] = number of ways from (0,0) to (i,j) with XOR == x
        dp1 = [[[0]*16 for _ in range(n)] for __ in range(m)]
        dp1[0][0][grid[0][0]] = 1
        
        # Build dp1 up to layer = mid
        for layer in range(mid):
            next_layer = layer + 1
            # i from max(0, next_layer-(n-1)) to min(m-1, next_layer)
            i_min = max(0, next_layer - (n-1))
            i_max = min(m-1, next_layer)
            for i in range(i_min, i_max+1):
                j = next_layer - i
                # compute dp1[i][j] from dp1[i-1][j] and dp1[i][j-1]
                val = grid[i][j]
                arr = dp1[i][j]
                # from top
                if i > 0:
                    prev = dp1[i-1][j]
                    for x in range(16):
                        cnt = prev[x]
                        if cnt:
                            arr[x ^ val] = (arr[x ^ val] + cnt) % MOD
                # from left
                if j > 0:
                    prev = dp1[i][j-1]
                    for x in range(16):
                        cnt = prev[x]
                        if cnt:
                            arr[x ^ val] = (arr[x ^ val] + cnt) % MOD
        
        # dp2[i][j][x] = number of ways from (i,j) to (m-1,n-1) with XOR == x
        # we'll build backwards from end down to layer = mid
        dp2 = [[[0]*16 for _ in range(n)] for __ in range(m)]
        dp2[m-1][n-1][grid[m-1][n-1]] = 1
        
        for layer in range(total, mid, -1):
            prev_layer = layer - 1
            i_min = max(0, prev_layer - (n-1))
            i_max = min(m-1, prev_layer)
            for i in range(i_min, i_max+1):
                j = prev_layer - i
                val = grid[i][j]
                arr = dp2[i][j]
                # from down (i+1,j)
                if i+1 < m:
                    nxt = dp2[i+1][j]
                    for x in range(16):
                        cnt = nxt[x]
                        if cnt:
                            arr[x ^ val] = (arr[x ^ val] + cnt) % MOD
                # from right (i,j+1)
                if j+1 < n:
                    nxt = dp2[i][j+1]
                    for x in range(16):
                        cnt = nxt[x]
                        if cnt:
                            arr[x ^ val] = (arr[x ^ val] + cnt) % MOD
        
        # Combine at layer = mid
        ans = 0
        layer = mid
        i_min = max(0, layer - (n-1))
        i_max = min(m-1, layer)
        for i in range(i_min, i_max+1):
            j = layer - i
            if j < 0 or j >= n:
                continue
            val = grid[i][j]
            a1 = dp1[i][j]
            a2 = dp2[i][j]
            # We need XOR1 ^ XOR2 ^ val == k  => XOR2 == XOR1 ^ val ^ k
            for x in range(16):
                cnt1 = a1[x]
                if cnt1:
                    need = x ^ val ^ k
                    cnt2 = a2[need]
                    if cnt2:
                        ans = (ans + cnt1 * cnt2) % MOD
        
        return ans