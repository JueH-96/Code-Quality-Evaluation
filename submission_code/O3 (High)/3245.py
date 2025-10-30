from bisect import bisect_left
from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        """
        1. Scan `s` once and collect every start index where `a` occurs (list `pos_a`)
           and every start index where `b` occurs (list `pos_b`).  
           Because we scan from left to right both lists are naturally sorted.
        
        2. For every index `i` in `pos_a` we need to know whether there exists an
           index `j` in `pos_b` with |j - i| ≤ k.  
           Since `pos_b` is sorted we can binary-search for the leftmost position
           that can satisfy the inequality (`j >= i - k`). If such a position exists
           and the actual j is also ≤ i + k, the condition holds.

        3. Collect all indices that satisfy the requirement and return them.
        """
        # helper to find all occurrences of a pattern (length ≤ 10) in s
        def find_occurrences(pat: str) -> List[int]:
            m = len(pat)
            res = []
            # last start index that allows full pattern
            limit = len(s) - m
            for i in range(limit + 1):
                if s[i:i+m] == pat:
                    res.append(i)
            return res
        
        pos_a = find_occurrences(a)
        pos_b = find_occurrences(b)
        
        # early exit if no `b` occurs
        if not pos_b:
            return []
        
        beautiful = []
        for i in pos_a:
            # j must be in [i-k, i+k]
            lo = i - k
            hi = i + k
            idx = bisect_left(pos_b, lo)   # first b-index ≥ lo
            if idx < len(pos_b) and pos_b[idx] <= hi:
                beautiful.append(i)
        
        return beautiful