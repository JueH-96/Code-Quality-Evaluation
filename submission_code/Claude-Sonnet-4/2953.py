class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import defaultdict
        
        coord_count = defaultdict(int)
        count = 0
        
        for x, y in coordinates:
            # For current point (x, y), find all points that have distance k
            # We need (x XOR x2) + (y XOR y2) = k
            # Try all possible values for (y XOR y2) from 0 to k
            for y_xor in range(k + 1):
                x_xor = k - y_xor
                # We need x2 such that x XOR x2 = x_xor
                # So x2 = x XOR x_xor
                x2 = x ^ x_xor
                y2 = y ^ y_xor
                
                if (x2, y2) in coord_count:
                    count += coord_count[(x2, y2)]
            
            coord_count[(x, y)] += 1
        
        return count