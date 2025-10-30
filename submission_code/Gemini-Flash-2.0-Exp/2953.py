class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = 0
        n = len(coordinates)
        for i in range(n):
            for j in range(i + 1, n):
                distance = (coordinates[i][0] ^ coordinates[j][0]) + (coordinates[i][1] ^ coordinates[j][1])
                if distance == k:
                    count += 1
        return count