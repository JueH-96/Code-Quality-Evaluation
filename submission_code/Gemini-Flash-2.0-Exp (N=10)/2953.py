class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = 0
        n = len(coordinates)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                distance = (x1 ^ x2) + (y1 ^ y2)
                if distance == k:
                    count += 1
        return count