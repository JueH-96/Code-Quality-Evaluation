import collections
from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        """
        Counts the number of pairs of points (i, j) with i < j such that
        the distance between coordinates[i] and coordinates[j] is k.
        Distance is defined as (x1 XOR x2) + (y1 XOR y2).

        Args:
            coordinates: A list of 2D integer coordinates [x, y].
            k: The target distance.

        Returns:
            The number of pairs (i, j) with i < j meeting the distance criteria.
        """
        count = 0
        # Use a dictionary to store the frequency of each point encountered so far.
        # The key is the tuple (x, y) and the value is its frequency.
        point_counts = collections.defaultdict(int)

        # Iterate through the points one by one.
        # For each point (x, y), we consider it as the second point in a pair (px, py).
        # We look up how many previous points (px, py) satisfy the distance condition.
        for x, y in coordinates:
            # The condition is (px XOR x) + (py XOR y) = k.
            # Let dx = px XOR x and dy = py XOR y.
            # We need dx + dy = k, where dx >= 0 and dy >= 0.
            # dx can take any integer value from 0 to k.
            # If dx = d, then dy must be k - d.
            # We need to find (px, py) such that:
            # px XOR x = d  => px = x XOR d
            # py XOR y = k - d => py = y XOR (k - d)

            # Iterate through all possible values 'd' for dx (px XOR x)
            for d in range(k + 1):
                # Calculate the required value for dy (py XOR y)
                dy = k - d

                # Calculate the coordinates (px, py) of the partner point we are looking for
                target_px = x ^ d
                target_py = y ^ dy

                # Check if this target partner point (target_px, target_py) has been seen before.
                # If it has, any occurrence of it forms a pair with the current point (x, y)
                # that satisfies the distance condition.
                # Add the number of occurrences of the target point to the total count.
                # defaultdict handles the case where the target point hasn't been seen yet by returning 0.
                count += point_counts[(target_px, target_py)]

            # After checking for partners for the current point (x, y), add it to the count of seen points.
            # This way, when we process subsequent points, (x, y) will be available as a potential partner.
            point_counts[(x, y)] += 1

        # Return the total number of pairs found.
        # This approach counts each pair (i, j) with i < j exactly once:
        # when the point with the larger index (say j) is processed, it looks up
        # previous points (with index i < j).
        return count