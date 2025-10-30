from typing import List
from collections import defaultdict

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        # Map to count occurrences of each point we've seen so far
        seen = defaultdict(int)
        count = 0
        
        # For each point, we look for prior points (x', y') such that
        # (x XOR x') + (y XOR y') == k.
        # If we fix d = (x XOR x'), then the remainder is k - d = (y XOR y').
        # Hence we iterate d from 0..k and compute the unique candidate point.
        for x, y in coordinates:
            # Try splitting k into dx + dy
            for dx in range(k + 1):
                dy = k - dx
                # The prior point must have x' = x XOR dx, y' = y XOR dy
                target = (x ^ dx, y ^ dy)
                count += seen.get(target, 0)
            
            # Record this point for future pairs
            seen[(x, y)] += 1
        
        return count