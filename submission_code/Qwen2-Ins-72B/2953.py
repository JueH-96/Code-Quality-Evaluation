from collections import Counter

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        counter = Counter()
        pairs = 0
        for x1, y1 in coordinates:
            for i in range(k + 1):
                x2 = i ^ x1
                y2 = (k - i) ^ y1
                pairs += counter[(x2, y2)]
            counter[(x1, y1)] += 1
        return pairs