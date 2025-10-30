from typing import List

class Solution:
    def getMaxFunctionValue(self, receiver: List[int], k: int) -> int:
        n = len(receiver)
        # Maximum power for binary lifting:
        # since k <= 1e10 < 2^34
        LOG = k.bit_length()
        
        # up[u][j]: the node reached from u after 2^j passes
        # sum_up[u][j]: sum of the receiver IDs in those 2^j passes
        up = [[0]*LOG for _ in range(n)]
        sum_up = [[0]*LOG for _ in range(n)]
        
        # Initialize the 2^0 = 1 step
        for u in range(n):
            v = receiver[u]
            up[u][0] = v
            sum_up[u][0] = v  # one pass adds v
        
        # Build binary lifting tables
        for j in range(1, LOG):
            for u in range(n):
                mid = up[u][j-1]
                up[u][j] = up[mid][j-1]
                sum_up[u][j] = sum_up[u][j-1] + sum_up[mid][j-1]
        
        ans = 0
        # For each starting x, do binary-lifted simulation of k passes
        for x in range(n):
            total = x
            cur = x
            rem = k
            # from highest bit down to 0
            b = 0
            while rem:
                if rem & 1:
                    total += sum_up[cur][b]
                    cur = up[cur][b]
                rem >>= 1
                b += 1
            if total > ans:
                ans = total
        
        return ans