class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = 0
        seen = {}
        for x, y in coordinates:
            count += seen.get((x, y), 0)
            for i in range(k + 1):
                seen[(x ^ i, y ^ (k - i))] = seen.get((x ^ i, y ^ (k - i)), 0) + 1
        return count