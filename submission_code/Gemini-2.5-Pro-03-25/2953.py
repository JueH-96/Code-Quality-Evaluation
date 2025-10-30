import collections
from typing import List

class Solution:
    """
    This class provides a method to count pairs of coordinates with a specific XOR distance.
    """
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        """
        Counts pairs (i, j) such that i < j and the XOR distance between
        coordinates[i] = (x_i, y_i) and coordinates[j] = (x_j, y_j) is equal to k.
        The XOR distance is defined as (x_i XOR x_j) + (y_i XOR y_j).

        Args:
            coordinates: A list of 2D integer coordinates, where coordinates[i] = [x_i, y_i].
            k: The target XOR distance.

        Returns:
            The number of pairs (i, j) satisfying the condition.

        Approach:
        We iterate through each point p_j = (x_j, y_j) in the `coordinates` list. For each point `p_j`, 
        we aim to find the number of previous points p_i = (x_i, y_i) (where the index `i` is less than `j`) 
        such that the XOR distance between `p_i` and `p_j` equals `k`.
        To do this efficiently, we use a hash map (specifically, `collections.Counter`) to store the 
        frequencies of points encountered so far during the iteration.

        The condition we need to satisfy is (x_i ^ x_j) + (y_i ^ y_j) = k.
        Let `x_xor = x_i ^ x_j` and `y_xor = y_i ^ y_j`. The condition becomes `x_xor + y_xor = k`.
        Since XOR results are non-negative, both `x_xor` and `y_xor` must be non-negative integers.
        We can iterate through all possible values for `x_xor` from 0 up to `k`. Let's call this value `val_x`.
        If `x_xor = val_x`, then `y_xor` must be equal to `k - val_x`. Let's call this `val_y`.
        
        From the properties of XOR (`a ^ b = c` implies `a ^ c = b`), if we are looking for a point `p_i = (x_i, y_i)` 
        such that `x_i ^ x_j = val_x` and `y_i ^ y_j = val_y`, then the coordinates of `p_i` must be:
        x_i = x_j ^ val_x
        y_i = y_j ^ val_y
        
        So, for each point `p_j` and for each possible split `k = val_x + val_y`, we calculate the 
        `target_coordinates = (x_j ^ val_x, y_j ^ val_y)`. We then look up how many times this 
        `target_coordinates` point has appeared previously (i.e., at an index `i < j`) using our frequency map. 
        The count stored in the map for `target_coordinates` tells us exactly how many such `p_i` exist. 
        We add this count to our total result.

        After processing point `p_j` (i.e., checking all possible splits of `k` and adding the counts of matching 
        previous points), we update the frequency map by incrementing the count for `p_j = (x_j, y_j)`. 
        This ensures that `p_j` can be matched with subsequent points in the list.

        Time Complexity: O(N * K), where N is the number of coordinates and K is the target distance k.
                       The outer loop iterates N times (once for each coordinate). The inner loop iterates K+1 times 
                       (for `val_x` from 0 to k). Operations inside the loop (XOR, addition, map lookup/update) 
                       take O(1) time on average.
        Space Complexity: O(N), in the worst case where all N points are distinct, the frequency map (`point_counts`) 
                        will store N entries.
        """

        # Initialize the count of valid pairs to 0.
        count = 0

        # Use collections.Counter to store the frequency of each point encountered so far.
        # The keys are tuples representing coordinates (x, y), and the values are their counts.
        # Example: point_counts[(1, 2)] = 3 means the point (1, 2) has appeared 3 times so far.
        point_counts = collections.Counter()

        # Iterate through each point p_j = (x_j, y_j) in the coordinates list.
        # Note: coordinates are given as List[List[int]], so coordinates[j] is [x_j, y_j].
        # We can unpack the list [x_j, y_j] directly in the loop.
        for x_j, y_j in coordinates:
            
            # For the current point (x_j, y_j), iterate through all possible values
            # for the x-component of the XOR distance, val_x.
            # val_x represents a possible value for (x_i ^ x_j). It can range from 0 to k.
            for val_x in range(k + 1):
                # Calculate the required value for the y-component of the XOR distance, val_y.
                # Since (x_i ^ x_j) + (y_i ^ y_j) = k, if x_i ^ x_j = val_x, then y_i ^ y_j must be k - val_x.
                val_y = k - val_x
                # Since val_x >= 0 and val_x <= k, val_y = k - val_x will always be non-negative (>= 0).

                # If a previous point (x_i, y_i) exists such that x_i ^ x_j = val_x and y_i ^ y_j = val_y,
                # then its coordinates must satisfy:
                # x_i = x_j ^ val_x
                # y_i = y_j ^ val_y
                
                # Calculate the coordinates (target_x, target_y) of this potential previous point p_i.
                target_x = x_j ^ val_x
                target_y = y_j ^ val_y

                # Look up the count of this target point (target_x, target_y) in our frequency map `point_counts`.
                # The value `point_counts[(target_x, target_y)]` represents the number of times the point 
                # (target_x, target_y) has appeared at indices i < current index j.
                # Each such previous occurrence forms a valid pair with the current point (x_j, y_j).
                # collections.Counter conveniently returns 0 if the key (target_x, target_y) is not found,
                # so we don't need an explicit check.
                count += point_counts[(target_x, target_y)]

            # After checking all possible splits of k (by varying val_x from 0 to k) and adding the counts
            # of matching previous points for the current point (x_j, y_j),
            # we update the frequency map by incrementing the count for the current point (x_j, y_j).
            # This makes it available for matching with points that appear later in the `coordinates` list.
            point_counts[(x_j, y_j)] += 1

        # Finally, return the total count of pairs found that satisfy the XOR distance condition.
        return count