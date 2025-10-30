import collections # Although not strictly required for this solution, it's good practice for complex scenarios.
from typing import List # Used for type hinting

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        """
        Calculates the total number of unique integer points covered by a list of intervals.

        The problem asks for the count of distinct integer points that fall within
        at least one of the given intervals [start_i, end_i].

        We can use a set to keep track of all the unique points covered. We iterate
        through each interval (car) provided in the input list `nums`. For each
        interval [start, end], we iterate through all the integer points from `start`
        to `end` (inclusive) and add each point to the set. Since sets only store
        unique elements, any point covered by multiple cars will only be added once.
        Finally, the size of the set gives the total number of unique points covered.

        Args:
            nums: A list of lists, where each inner list [start_i, end_i] represents
                  the start and end points (inclusive) of the space covered by a car
                  on a number line.

        Returns:
            The total count of distinct integer points covered by any car.

        Example:
            nums = [[3,6],[1,5],[4,7]]
            Car 1 covers: 3, 4, 5, 6
            Car 2 covers: 1, 2, 3, 4, 5
            Car 3 covers: 4, 5, 6, 7
            Set of covered points: {1, 2, 3, 4, 5, 6, 7}
            Size of set: 7
        """
        
        covered_points = set() # Initialize an empty set to store unique covered points

        # Iterate through each car's interval [start, end] in the input list
        for start, end in nums:
            # Iterate through each integer point from start to end (inclusive)
            # range(start, end + 1) generates numbers start, start+1, ..., end
            for point in range(start, end + 1):
                # Add the current point to the set.
                # If the point is already in the set, add() does nothing.
                covered_points.add(point)

        # The number of unique points covered is simply the size (length) of the set.
        return len(covered_points)