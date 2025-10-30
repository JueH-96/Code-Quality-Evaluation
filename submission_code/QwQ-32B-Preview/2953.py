from typing import List
from collections import defaultdict

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = 0
        freq = defaultdict(int)
        
        for coord in coordinates:
            x, y = coord
            # For each possible a from 0 to k
            for a in range(k + 1):
                b = k - a
                x_prime = x ^ a
                y_prime = y ^ b
                # Increment count by the frequency of (x', y')
                count += freq[(x_prime, y_prime)]
            # Increment the frequency of the current point
            freq[(x, y)] += 1
        
        return count