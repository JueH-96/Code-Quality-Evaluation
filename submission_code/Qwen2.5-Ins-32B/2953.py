from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        from collections import defaultdict
        count = defaultdict(int)
        result = 0
        
        for x, y in coordinates:
            for i in range(k + 1):
                xi = x ^ i
                yi = y ^ (k - i)
                result += count[(xi, yi)]
            count[(x, y)] += 1
        
        return result