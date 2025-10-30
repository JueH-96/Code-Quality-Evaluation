class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = 0
        for i in range(len(coordinates)):
            for j in range(i+1, len(coordinates)):
                xor_distance = (coordinates[i][0] ^ coordinates[j][0]) + (coordinates[i][1] ^ coordinates[j][1])
                if xor_distance == k:
                    count += 1
        return count