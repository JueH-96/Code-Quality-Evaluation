from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = 0
        n = len(coordinates)
        
        # Iterate over all pairs of points
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate the distance between the two points
                distance = (coordinates[i][0] ^ coordinates[j][0]) + (coordinates[i][1] ^ coordinates[j][1])
                
                # If the distance is equal to k, increment the count
                if distance == k:
                    count += 1
                    
        return count