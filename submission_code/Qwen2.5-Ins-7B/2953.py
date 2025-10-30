from collections import defaultdict

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = 0
        seen = defaultdict(int)
        for x, y in coordinates:
            for i in range(k + 1):
                j = k - i
                count += seen[(x ^ i, y ^ j)]
            seen[(x, y)] += 1
        return count