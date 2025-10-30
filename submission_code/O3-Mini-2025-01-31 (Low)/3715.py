from typing import List
import bisect

class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Preprocessing
        # sort segments by left endpoint
        coins.sort(key=lambda x: x[0])
        n = len(coins)
        # Create arrays: L, R, C, and preSum (the prefix sum of full segments contributions)
        L = []
        R = []
        C = []
        full = []  # coins in full segment: (r - l + 1) * c
        for seg in coins:
            li, ri, ci = seg
            L.append(li)
            R.append(ri)
            C.append(ci)
            full.append((ri - li + 1) * ci)
            
        preSum = [0] * n
        preSum[0] = full[0]
        for i in range(1, n):
            preSum[i] = preSum[i-1] + full[i]
            
        # Function F(x): returns total coins in all bags with coord <= x.
        # If x < first segment's left, F(x)=0.
        def F(x: int) -> int:
            if x < L[0]:
                return 0
            # find rightmost segment whose r <= x
            idx = bisect.bisect_right(R, x) - 1 
            total = preSum[idx] if idx >= 0 else 0
            # Now, check if there is a next segment (idx+1) which starts at or before x and x falls in it.
            if idx + 1 < n and L[idx+1] <= x:
                # We are in segment idx+1, but x may exceed R[idx+1] so we only add up to R.
                addition = min(x, R[idx+1]) - L[idx+1] + 1
                total += addition * C[idx+1]
            return total
        
        # Prepare candidate starting positions.
        # As argued, candidate x are either equal to L[i] (the beginning of a coin segment)
        # or x = R[i] - k + 1 if that is inside the segment (i.e. >= L[i]).
        candidates = set()
        for i in range(n):
            # ensure candidate x is at least 1 (bags are labelled from 1)
            if L[i] >= 1:
                candidates.add(L[i])
            # candidate that is the last position from which the segment's coins can be collected.
            cand = R[i] - k + 1
            if cand >= L[i] and cand >= 1:
                candidates.add(cand)
                
        # It is possible that none of these positions is valid if, say, k is huge
        # In that case, picking x = 1 is a candidate.
        if not candidates:
            candidates.add(1)
        
        best = 0
        # (To speed up a little bit if needed, we can sort candidates)
        for x in candidates:
            # The window is [x, x+k-1]
            val = F(x + k - 1) - F(x - 1)
            if val > best:
                best = val
        return best