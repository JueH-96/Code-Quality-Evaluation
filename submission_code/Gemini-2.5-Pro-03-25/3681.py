from typing import List

class Solution:
    """
    Finds the maximum area of an axis-parallel rectangle formed by four points 
    from the input list, such that the rectangle contains no other points from 
    the list inside or on its border.
    """
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        """
        :param points: A list of points, where each point is [x, y]. Coordinates are integers.
        :return: The maximum area of such a rectangle, or -1 if no such rectangle exists.
        """
        N = len(points)
        # A rectangle requires 4 corner points. If there are fewer than 4 points,
        # no rectangle can be formed.
        if N < 4:
            return -1

        # Use a set of tuples for efficient point lookups (O(1) average time).
        # Tuples are hashable and can be stored in a set, unlike lists.
        point_set = set()
        for p in points:
            point_set.add(tuple(p))

        max_area = 0  # Initialize max_area to 0. We will track the maximum positive area found.

        # Iterate through all distinct pairs of points (p1, p2) using their indices i and j.
        # We consider each pair as potential diagonally opposite corners of a rectangle.
        for i in range(N):
            for j in range(i + 1, N):
                x1, y1 = points[i]
                p1_tuple = (x1, y1)
                x2, y2 = points[j]
                p2_tuple = (x2, y2)

                # Check if p1 and p2 can form a valid diagonal for an axis-parallel rectangle.
                # They must have different x and different y coordinates 
                # to form a rectangle with non-zero area.
                if x1 == x2 or y1 == y2:
                    continue # Skip if points are on the same horizontal or vertical line.

                # Calculate the coordinates of the other two potential corners required to form the rectangle.
                # For a rectangle with diagonal corners (x1, y1) and (x2, y2), the other corners are (x1, y2) and (x2, y1).
                p3_tuple = (x1, y2)
                p4_tuple = (x2, y1)

                # Verify if these two potential corners actually exist in the input set of points.
                # Use the point_set for fast checking.
                if p3_tuple in point_set and p4_tuple in point_set:
                    # If both points exist, we have found four points from the input list
                    # that form an axis-parallel rectangle.
                    # The four corners are: p1_tuple, p2_tuple, p3_tuple, p4_tuple.
                    
                    # Define the rectangle boundaries using min/max coordinates for the emptiness check.
                    min_x = min(x1, x2)
                    max_x = max(x1, x2)
                    min_y = min(y1, y2)
                    max_y = max(y1, y2)

                    # Check the emptiness condition: The rectangle must not contain any *other* point
                    # from the input list strictly inside it or on its boundary edges (besides the four corners).
                    is_empty = True
                    # Store the four corner points in a set for efficient checking during the emptiness test.
                    corners = {p1_tuple, p2_tuple, p3_tuple, p4_tuple}
                    
                    # Iterate through all given points (indexed by k) to check if any point violates the emptiness condition.
                    for k in range(N):
                        # Get the k-th point's coordinates and its tuple representation.
                        px, py = points[k]
                        pk_tuple = (px, py)

                        # Check if the point pk lies within the rectangle boundary (inclusive).
                        # A point (px, py) is inside or on the border if min_x <= px <= max_x and min_y <= py <= max_y.
                        if min_x <= px <= max_x and min_y <= py <= max_y:
                            # If pk is inside or on the boundary, it must be one of the four corners.
                            # If it's found within the boundary but is NOT one of the identified corners,
                            # then the rectangle is not "empty" according to the problem definition.
                            if pk_tuple not in corners:
                                is_empty = False
                                break # Exit the inner loop early, as we've found an violating point.
                    
                    # If the inner loop completed without finding any violating points, the rectangle is empty.
                    if is_empty:
                        # Calculate the area of this valid empty rectangle.
                        # The area is guaranteed to be positive because we checked x1 != x2 and y1 != y2 earlier.
                        area = abs(x2 - x1) * abs(y2 - y1)
                        # Update the maximum area found so far across all valid rectangles.
                        max_area = max(max_area, area)

        # After checking all possible pairs of points and potential rectangles:
        # If max_area is still 0, it implies that no valid empty rectangle was found.
        # In this case, return -1 as required by the problem statement.
        # Otherwise, return the computed maximum area.
        return max_area if max_area > 0 else -1