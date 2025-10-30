class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = 0
        seen = {}
        
        for x, y in coordinates:
            # For each new point, we check all possible values a where a + b = k
            # For each value of a, we look for points that would give us distance k
            for a in range(k + 1):
                b = k - a
                # If (x, y) XOR (p, q) = k, then (p, q) = (x ^ a, y ^ b) where a + b = k
                count += seen.get((x ^ a, y ^ b), 0)
            
            # Add current point to our seen dictionary
            seen[(x, y)] = seen.get((x, y), 0) + 1
        
        return count