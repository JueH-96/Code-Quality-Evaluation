from collections import Counter
from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        # Build frequency dictionary for points
        freq = Counter(tuple(pt) for pt in coordinates)
        ans = 0
        
        # Iterate over all possible (dx, dy) pairs such that dx + dy == k 
        # (dx and dy are the results of XOR between corresponding coordinates)
        # Note: If k == 0, then only (dx, dy) = (0,0) qualifies.
        for dx in range(0, k + 1):
            dy = k - dx
            # For (dx, dy) == (0,0), we need to count pairs from the same point group
            if dx == 0 and dy == 0:
                for p in freq:
                    count = freq[p]
                    ans += count * (count - 1) // 2
            else:
                # For any other (dx,dy), for each point p, check if the "partner"
                # point q = (p[0] XOR dx, p[1] XOR dy) exists.
                # To avoid double counting (since the mapping is an involution),
                # we only count the pair if p < q (using lexicographical order)
                for p in freq:
                    q = (p[0] ^ dx, p[1] ^ dy)
                    if q in freq and p < q:
                        ans += freq[p] * freq[q]
        return ans