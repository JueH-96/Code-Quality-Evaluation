from typing import List

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        # Precompute powers of 4 up to >1e9
        pows = [1]
        while pows[-1] <= 10**9:
            pows.append(pows[-1] * 4)
        # Now pows[k] = 4^k for k=0,1,...,K, and pows[K] > 1e9
        
        total_ops = 0
        for l, r in queries:
            # Compute S = sum_{x=l..r} (floor(log4(x)) + 1)
            S = 0
            # For each interval [4^k, 4^{k+1}-1]
            # floor(log4(x)) = k  iff x in [pows[k], pows[k+1]-1]
            for k in range(len(pows) - 1):
                lo = pows[k]
                hi = pows[k+1] - 1
                if hi < l or lo > r:
                    continue
                a = max(l, lo)
                b = min(r, hi)
                cnt = b - a + 1
                # Each such x contributes (k+1)
                S += cnt * (k + 1)
            # Each operation gives two "reductions"; we need ceil(S / 2)
            ops = (S + 1) // 2
            total_ops += ops
        
        return total_ops