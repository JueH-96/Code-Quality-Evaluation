class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count_map = {}
        res = 0
        for x, y in coordinates:
            for a in range(k + 1):
                b = k - a
                x1 = x ^ a
                y1 = y ^ b
                res += count_map.get((x1, y1), 0)
            count_map[(x, y)] = count_map.get((x, y), 0) + 1
        return res