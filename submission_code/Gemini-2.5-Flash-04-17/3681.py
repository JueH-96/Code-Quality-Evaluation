import itertools
from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        """
        Finds the maximum area of an axis-aligned rectangle formed by four points
        from the input list, such that the rectangle contains no other points
        inside or on its border.

        Args:
            points: A list of points, where points[i] = [x_i, y_i].

        Returns:
            The maximum area of such a rectangle, or -1 if no such rectangle is possible.
        """
        n = len(points)
        # A rectangle requires at least 4 points.
        if n < 4:
            return -1

        # Convert points to a set of tuples for efficient O(1) average time lookups.
        # This is primarily used in the emptiness check to quickly determine if a point
        # is one of the current rectangle's corners.
        point_set = set(tuple(p) for p in points)

        max_area = -1 # Initialize max area to -1, as required if no valid rectangle is found.

        # Iterate through all combinations of 4 distinct points from the input list.
        # Since the input points are guaranteed to be unique, itertools.combinations
        # automatically gives us combinations of distinct points.
        # The number of combinations is N choose 4, which is small for N <= 10.
        # For N=10, combinations(10, 4) = 10*9*8*7 / (4*3*2*1) = 210.
        for p1, p2, p3, p4 in itertools.combinations(points, 4):
            # Extract coordinates of the four chosen points.
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4

            # Collect the x and y coordinates of the four chosen points into sets
            # to easily count distinct values.
            xs = {x1, x2, x3, x4}
            ys = {y1, y2, y3, y4}

            # Check if these 4 points *could* form the corners of an axis-aligned rectangle.
            # An axis-aligned rectangle formed by 4 points must have exactly two distinct x-coordinates
            # and exactly two distinct y-coordinates among its four corners.
            if len(xs) == 2 and len(ys) == 2:
                # Get the minimum and maximum x and y values that define the rectangle's bounds.
                # Sorting the set {x_val1, x_val2} gives [x_min, x_max].
                x_vals = sorted(list(xs))
                y_vals = sorted(list(ys))
                x_min, x_max = x_vals[0], x_vals[1]
                y_min, y_max = y_vals[0], y_vals[1]

                # Ensure non-degenerate rectangle (width and height > 0).
                # This check is implicitly covered by the len(xs)==2 and len(ys)==2 conditions
                # on unique points (if len(xs) was 2, x_min must be different from x_max as points are unique),
                # but doesn't harm.
                # if x_min == x_max or y_min == y_max: # Redundant given constraints and earlier checks
                #    continue

                # Verify that the selected four points are *exactly* the four corners
                # defined by the found x_min, x_max, y_min, y_max.
                # The four corners of an axis-aligned rectangle with x in {x_min, x_max} and y in {y_min, y_max}
                # must be (x_min, y_min), (x_min, y_max), (x_max, y_min), and (x_max, y_max).
                required_corners = {
                    (x_min, y_min),
                    (x_min, y_max),
                    (x_max, y_min),
                    (x_max, y_max)
                }
                # Convert the chosen 4 points into a set of tuples for easy comparison.
                given_points_set = {tuple(p1), tuple(p2), tuple(p3), tuple(p4)}

                # If the selected points perfectly match the corners of an axis-aligned rectangle:
                if required_corners == given_points_set:
                    # We've found a geometric rectangle formed by 4 input points.
                    # Now, check the condition: "Does not contain any other point inside or on its border."
                    is_empty = True # Assume it's empty initially.

                    # Iterate through *all* original points to check for intruders.
                    # This loop runs N times.
                    for other_point in points:
                        other_x, other_y = other_point
                        other_point_tuple = tuple(other_point)

                        # If the current 'other_point' is one of the 4 points forming the rectangle,
                        # it's not an "other" point we need to check for emptiness violation. Skip it.
                        if other_point_tuple in given_points_set:
                            continue

                        # Check if this 'other_point' lies within the bounding box of the rectangle
                        # [x_min, x_max] x [y_min, y_max]. This includes points strictly inside
                        # and points exactly on any of the four sides (border).
                        # The condition x_min <= px <= x_max and y_min <= py <= y_max covers points on the boundary (x=x_min or x_max, y=y_min or y_max)
                        # and points strictly inside (x_min < px < x_max and y_min < py < y_max).
                        if x_min <= other_x <= x_max and y_min <= other_y <= y_max:
                            # Found an 'other' point inside or on the border.
                            # This rectangle is NOT empty according to the problem definition.
                            is_empty = False # Mark as not empty.
                            break # No need to check further for this specific rectangle.

                    # If the inner loop completed and 'is_empty' is still True, it means no
                    # other points were found inside or on the border.
                    if is_empty:
                        # This is a valid empty rectangle. Calculate its area.
                        area = (x_max - x_min) * (y_max - y_min)
                        # Update the maximum area found so far if this rectangle's area is larger.
                        max_area = max(max_area, area)

        # After checking all combinations of 4 points, return the maximum area found.
        # If no valid empty rectangle was found, max_area remains its initial value of -1.
        return max_area