class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import Counter
        
        # Frequency dictionary for all points
        freq = Counter(tuple(pt) for pt in coordinates)
        
        # Special case: if k == 0, the distance is 0 only if points are identical,
        # so we just sum up combinations of identical points.
        if k == 0:
            ans = 0
            for count in freq.values():
                ans += count * (count - 1) // 2
            return ans
        
        # For k > 0, use the formula:
        #   distance((x1,y1),(x2,y2)) = (x1 XOR x2) + (y1 XOR y2) = k
        # We will iterate over each distinct (x,y) and each possible a in [0..k],
        # let b = k - a. Then we look for the point (x',y') = (x XOR a, y XOR b).
        # To avoid double counting, only count if (x',y') >= (x,y) in lex order.
        
        ans = 0
        items = list(freq.items())  # [((x,y), count)]
        
        for (x, y), count1 in items:
            for a in range(k + 1):
                b = k - a
                x2 = x ^ a
                y2 = y ^ b
                
                count2 = freq.get((x2, y2), 0)
                if count2 == 0:
                    continue
                
                # Compare (x2, y2) with (x, y) in lexical order
                if (x2 < x) or (x2 == x and y2 < y):
                    # Already counted when we visited (x2, y2), skip
                    continue
                
                if x2 == x and y2 == y:
                    # This would imply a=0 and b=0 => but k>0, so no contribution
                    continue
                else:
                    # Count cross pairs
                    ans += count1 * count2
        
        return ans