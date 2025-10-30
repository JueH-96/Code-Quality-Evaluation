class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        chosen = set()   # numbers already taken
        total  = 0
        cur    = 1       # current candidate number
        
        # keep taking numbers with the smallest possible value
        while len(chosen) < n:
            # it is safe to pick `cur` iff its complement (k - cur)
            # is NOT already in the array.  For cur >= k the complement
            # is non-positive, so the condition is automatically true.
            if (k - cur) not in chosen:
                chosen.add(cur)
                total += cur
            cur += 1
        
        return total