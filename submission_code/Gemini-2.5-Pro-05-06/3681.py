from typing import List

class Solution:
  def maxRectangleArea(self, points: List[List[int]]) -> int:
    N = len(points)
    if N < 4:
        return -1

    # Store points in a set for quick lookup.
    # Points are [x,y] lists, convert to tuples to be hashable.
    point_set = set()
    for p_list in points:
        point_set.add(tuple(p_list))

    max_area = 0

    # Iterate over all unique pairs of points (P_i, P_j)
    # These pairs are treated as potential diagonals of a rectangle.
    for i in range(N):
        for j in range(i + 1, N): # Ensure (i,j) is a unique pair, P_i != P_j
            x_i, y_i = points[i]
            x_j, y_j = points[j]

            # P_i and P_j can form a diagonal of a rectangle
            # if their x-coordinates are different and y-coordinates are different.
            # This ensures the rectangle would have a positive area.
            if x_i == x_j or y_i == y_j:
                continue

            # The other two potential corners of the rectangle would be (x_i, y_j) and (x_j, y_i).
            # Let point1 be (x_i, y_i) (points[i])
            # Let point2 be (x_j, y_j) (points[j])
            # Then potential point3 is (x_i, y_j) and potential point4 is (x_j, y_i).
            
            # Check if these two other points exist in the input set.
            if (x_i, y_j) in point_set and (x_j, y_i) in point_set:
                # All four corners are present in the input `points`.
                # This forms a candidate rectangle.
                # The four corners are (x_i,y_i), (x_j,y_j), (x_i,y_j), (x_j,y_i).
                
                # Determine the minimum and maximum x and y coordinates for the bounding box.
                min_coord_x = min(x_i, x_j)
                max_coord_x = max(x_i, x_j)
                min_coord_y = min(y_i, y_j)
                max_coord_y = max(y_i, y_j)
                
                current_area = (max_coord_x - min_coord_x) * (max_coord_y - min_coord_y)
                
                # Check the "empty" condition: no other point (from input `points`)
                # should be strictly inside this rectangle or on its border
                # (excluding the four corners themselves).
                # We achieve this by counting how many points from the input `points`
                # lie within or on the boundary of the rectangle 
                # [min_coord_x, max_coord_x] x [min_coord_y, max_coord_y].
                # This count must be exactly 4 (for the four corners).
                
                count_points_in_bbox = 0
                # Iterate through all original points to check this condition.
                for k_idx in range(N): 
                    px, py = points[k_idx]
                    if min_coord_x <= px <= max_coord_x and \
                       min_coord_y <= py <= max_coord_y:
                        count_points_in_bbox += 1
                
                if count_points_in_bbox == 4:
                    # This is a valid empty rectangle. Update max_area.
                    max_area = max(max_area, current_area)

    if max_area == 0:
        # No valid rectangle with positive area was found.
        # (Positive area is guaranteed by `x_i != x_j and y_i != y_j` condition)
        return -1
    else:
        return max_area