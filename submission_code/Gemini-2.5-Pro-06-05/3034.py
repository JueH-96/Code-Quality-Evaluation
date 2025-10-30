from typing import List

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        """
        Calculates the number of unique integer points covered by a set of intervals.

        Args:
            nums: A list of lists, where each inner list [start, end] represents
                  an interval on the number line.

        Returns:
            The total number of unique integer points covered by any interval.
        """
        
        # A set is used to store the unique points covered by the cars.
        # This automatically handles overlaps between car parking spaces.
        covered_points = set()

        # Iterate through each car's parking interval [start, end].
        for start, end in nums:
            # Add all integer points from start to end (inclusive) to the set.
            # The range(start, end + 1) generates numbers from start up to end.
            for point in range(start, end + 1):
                covered_points.add(point)

        # The number of unique covered points is the size of the set.
        return len(covered_points)