class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import defaultdict
        
        count = 0
        seen = defaultdict(int)
        
        for x, y in coordinates:
            for i in range(k + 1):
                x_target = x ^ i
                y_target = y ^ (k - i)
                count += seen[(x_target, y_target)]
            
            seen[(x, y)] += 1
        
        return count