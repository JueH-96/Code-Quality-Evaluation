class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import defaultdict
        count = 0
        point_count = defaultdict(int)
        
        for x1, y1 in coordinates:
            for xor_x in range(k + 1):
                xor_y = k - xor_x
                if xor_y < 0:
                    continue
                    
                # Find x2, y2 that satisfy:
                # x1 ^ x2 = xor_x and y1 ^ y2 = xor_y
                x2 = x1 ^ xor_x
                y2 = y1 ^ xor_y
                
                count += point_count[(x2, y2)]
                
            point_count[(x1, y1)] += 1
            
        return count