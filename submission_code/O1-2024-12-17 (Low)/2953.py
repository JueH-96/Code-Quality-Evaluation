class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import defaultdict
        
        freq = defaultdict(int)
        pairs = 0
        
        # For each point encountered, we look up how many points seen so far
        # could form the XOR-distance = k with that point.
        # distance(x1, y1, x2, y2) = (x1 XOR x2) + (y1 XOR y2) = k.
        # For a fixed (x, y), if (x1 XOR x2) = r and (y1 XOR y2) = s, then r + s = k.
        # We iterate over all r from 0..k, let s = k-r, and check freq[x^r, y^s].
        
        for x, y in coordinates:
            for r in range(k + 1):
                s = k - r
                x2 = x ^ r
                y2 = y ^ s
                pairs += freq[(x2, y2)]
            
            # Now add the current point to the frequency map
            freq[(x, y)] += 1
        
        return pairs