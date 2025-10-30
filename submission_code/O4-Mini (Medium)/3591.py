from typing import List

class Solution:
    def shiftDistance(self, s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
        # Build prefix sums for nextCost and previousCost
        # nextPrefix[i] = sum of nextCost[0..i-1]
        # prevPrefix[i] = sum of previousCost[0..i-1]
        N = 26
        nextPrefix = [0] * (N + 1)
        prevPrefix = [0] * (N + 1)
        for i in range(N):
            nextPrefix[i+1] = nextPrefix[i] + nextCost[i]
            prevPrefix[i+1] = prevPrefix[i] + previousCost[i]
        
        total = 0
        for cs, ct in zip(s, t):
            a = ord(cs) - ord('a')
            b = ord(ct) - ord('a')
            if a == b:
                continue
            
            # forward distance: shift next from a to b
            # steps = (b - a + 26) % 26
            # cost = sum of nextCost[x] for x in [a .. b-1] mod 26
            if a < b:
                cost_next = nextPrefix[b] - nextPrefix[a]
            else:
                # wrap around
                cost_next = (nextPrefix[N] - nextPrefix[a]) + nextPrefix[b]
            
            # backward distance: shift previous from a to b
            # steps = (a - b + 26) % 26
            # cost = sum of previousCost[x] for x in [b+1 .. a] mod 26
            l = (b + 1) % N
            r = a
            if l <= r:
                cost_prev = prevPrefix[r+1] - prevPrefix[l]
            else:
                cost_prev = (prevPrefix[N] - prevPrefix[l]) + prevPrefix[r+1]
            
            total += min(cost_next, cost_prev)
        
        return total