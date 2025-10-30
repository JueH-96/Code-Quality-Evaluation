import collections
from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        count = 0
        # freq will store the count of each unique point (x, y) encountered so far.
        # Using collections.defaultdict(int) simplifies handling new keys,
        # as accessing a non-existent key will return 0 (default value for int).
        freq = collections.defaultdict(int)

        # Iterate through each point (x1, y1) in the coordinates list.
        # For each point, we consider it as the 'current' point (p1)
        # and look for 'previous' points (p2) that satisfy the distance condition.
        for x1, y1 in coordinates:
            # The distance formula is (x1 XOR x2) + (y1 XOR y2) = k.
            # Let dx = x1 XOR x2 and dy = y1 XOR y2.
            # We need dx + dy = k.
            # Since dx and dy are results of XOR operations, they are non-negative.
            # Also, since they sum to k, both dx and dy must be less than or equal to k.
            # We iterate through all possible integer values for dx from 0 to k.
            for dx in range(k + 1):
                dy = k - dx # Calculate the corresponding dy

                # If x1 XOR x2 = dx, then x2 = x1 XOR dx. (Property of XOR: A^B=C => A^C=B)
                # Similarly, if y1 XOR y2 = dy, then y2 = y1 XOR dy.
                target_x2 = x1 ^ dx
                target_y2 = y1 ^ dy
                
                # Check how many times the point (target_x2, target_y2)
                # has appeared among the points processed *before* the current (x1, y1).
                # These are the pairs (p_previous, p_current) that satisfy the condition.
                count += freq[(target_x2, target_y2)]
            
            # After checking all possible pairs with the current point (x1, y1),
            # add (x1, y1) to the frequency map so it can be considered
            # for subsequent points as a 'previous' point.
            freq[(x1, y1)] += 1
            
        return count