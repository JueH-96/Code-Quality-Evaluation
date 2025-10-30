from collections import Counter

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = 0
        coord_counter = Counter()
        for x, y in coordinates:
            for dx in range(k+1):
                dy = k - dx
                count += coord_counter[(x^dx, y^dy)]
            coord_counter[(x, y)] += 1
        return count