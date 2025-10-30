from typing import List
import math

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:

        def get_boundary_distance(p, side):
            """Calculates the cumulative distance of a point from (0,0) along the boundary."""
            x, y = p
            # We traverse the boundary from (0,0) counter-clockwise:
            # (0,0) -> (side,0) -> (side,side) -> (0,side) -> (0,0)
            if y == 0: # Bottom edge (0,0) to (side,0)
                return x
            elif x == side: # Right edge (side,0) to (side,side)
                return side + y
            elif y == side: # Top edge (side,side) to (0,side)
                # Distance decreases as x goes from side to 0
                return 2 * side + (side - x)
            elif x == 0: # Left edge (0,side) to (0,0)
                # Distance decreases as y goes from side to 0
                return 3 * side + (side - y)
            else:
                 # This case should not be reached based on problem constraints
                return -1 # Should ideally not happen

        def manhattan_distance(p1, p2):
            """Calculates the Manhattan distance between two points."""
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        # Sort points by boundary distance to order them along the perimeter
        sorted_points = sorted(points, key=lambda p: get_boundary_distance(p, side))
        n = len(sorted_points)
        
        # Create an extended list of points to easily handle wrap-around when selecting points
        # along the boundary. The list is points + points.
        extended_points = sorted_points + sorted_points

        def check(D):
            """
            Checks if it is possible to select k points from the boundary
            such that the minimum Manhattan distance between any two selected points is at least D.
            Uses a greedy strategy by iterating through each possible starting point
            and selecting the next available point with the smallest boundary distance
            that satisfies the minimum distance D requirement.
            """
            # Try each point in the original list as the starting point
            for start_index in range(n):
                current_extended_index = start_index
                count = 1
                success = True # Flag to track if we successfully selected k points

                # Greedily select k-1 additional points
                for _ in range(k - 1):
                    # We need to find the smallest index `next_extended_index`
                    # in the extended list, strictly greater than the current index,
                    # such that the Manhattan distance from the point at `current_extended_index`
                    # to the point at `next_extended_index` is at least D.
                    # We search within one full cycle ahead relative to the current point:
                    # indices from current_extended_index + 1 up to current_extended_index + n.
                    
                    bs_low = current_extended_index + 1
                    bs_high = current_extended_index + n
                    
                    next_extended_index = -1 # Stores the smallest valid index found in the search range

                    # Binary search for the smallest `next_extended_index`
                    # in the range [bs_low, bs_high] that satisfies the distance condition.
                    search_result_index = -1
                    temp_low = bs_low
                    temp_high = bs_high

                    while temp_low <= temp_high:
                        mid_extended_index = (temp_low + temp_high) // 2
                        if manhattan_distance(extended_points[current_extended_index], extended_points[mid_extended_index]) >= D:
                            search_result_index = mid_extended_index # mid is a potential next point, try earlier ones
                            temp_high = mid_extended_index - 1
                        else:
                            temp_low = mid_extended_index + 1 # mid is too close, need a larger index

                    next_extended_index = search_result_index

                    if next_extended_index == -1:
                        # If no point is found within the search range [current_extended_index + 1, current_extended_index + n]
                        # that is at least distance D away, we cannot select k points starting from this point with this D.
                        success = False
                        break # Exit the inner loop (selecting k-1 points)

                    # Move to the next selected point
                    current_extended_index = next_extended_index
                    count += 1

                # After attempting to select k points, check if successful
                # If successful, we selected indices j_0, j_1, ..., j_{k-1} in the extended list,
                # where j_0 = start_index, and j_0 < j_1 < ... < j_{k-1}.
                # For these k points to be distinct in the original list, the indices j_m % n must be distinct.
                # The condition `current_extended_index < start_index + n` guarantees that all j_0, ..., j_{k-1}
                # are within the first cycle relative to start_index (indices < start_index + n),
                # which implies their modulo n values are distinct.
                # We also need to check the wrap-around distance from the last selected point (index j_{k-1})
                # to the next instance of the first selected point (which is at index j_0 + n).
                if success and current_extended_index < start_index + n:
                    # Check the distance between the last point and the point corresponding to the start point in the next cycle
                    if manhattan_distance(extended_points[current_extended_index], extended_points[start_index + n]) >= D:
                        # Found a valid selection of k distinct points with minimum distance >= D
                        return True

            # If no starting point allowed selecting k points with minimum distance D, return False
            return False

        # Binary search for the maximum possible minimum distance D
        # The lower bound for D is 0.
        # The upper bound for D is the maximum possible Manhattan distance between any two points on the boundary.
        # The points farthest apart could be corners, e.g., (0,0) and (side, side).
        # Distance = |0 - side| + |0 - side| = 2 * side.
        low_D = 0
        high_D = 2 * side
        ans_D = 0 # Initialize with minimum possible answer

        while low_D <= high_D:
            mid_D = (low_D + high_D) // 2
            if check(mid_D):
                # It is possible to select k points with minimum distance mid_D
                ans_D = mid_D # mid_D is achievable, store it as a potential answer
                low_D = mid_D + 1 # Try a larger minimum distance
            else:
                # It is not possible to select k points with minimum distance mid_D
                high_D = mid_D - 1 # mid_D is too large, try a smaller distance

        return ans_D