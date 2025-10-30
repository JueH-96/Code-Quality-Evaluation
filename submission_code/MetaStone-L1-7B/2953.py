from collections import defaultdict

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = 0
        point_map = defaultdict(int)
        for x, y in coordinates:
            for a in range(k + 1):
                b = k - a
                target_x = x ^ a
                target_y = y ^ b
                count += point_map.get((target_x, target_y), 0)
            point_map[(x, y)] += 1
        return count