import collections
from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        """
        Counts pairs of points (i, j) with i < j where the XOR distance equals k.
        XOR distance is (x1^x2) + (y1^y2).

        This solution uses a hash map to achieve O(n*k) time complexity.
        """
        # A hash map to store the frequency of each point encountered so far.
        # collections.defaultdict(int) initializes new keys with a value of 0.
        seen_points = collections.defaultdict(int)
        count = 0

        for x1, y1 in coordinates:
            # For the current point (x1, y1), we need to find a previously seen
            # point (x2, y2) such that (x1^x2) + (y1^y2) = k.
            #
            # Let a = x1^x2 and b = y1^y2. The equation is a + b = k.
            # From XOR properties: x2 = x1^a and y2 = y1^b.
            #
            # We can iterate through all possible values of 'a' from 0 to k.
            for a in range(k + 1):
                # 'b' is determined by the value of 'a' and k.
                b = k - a
                
                # Calculate the coordinates of the target partner point we are looking for.
                target_x = x1 ^ a
                target_y = y1 ^ b
                
                # Add the number of times we've seen this target point to our count.
                # If (target_x, target_y) is not in seen_points, defaultdict returns 0.
                count += seen_points[(target_x, target_y)]
            
            # After checking for pairs, add the current point to our map
            # to make it available for subsequent points.
            seen_points[(x1, y1)] += 1
            
        return count