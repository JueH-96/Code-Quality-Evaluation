from typing import List

class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        max_area = -1
        
        # Convert list of lists to a set of tuples for O(1) average time lookups.
        # This allows for quick checking of point existence.
        point_set = set(tuple(p) for p in points)
        
        n = len(points)
        
        # Iterate over all possible pairs of points to find potential diagonals.
        # Let p1 = (x1, y1) and p2 = (x2, y2).
        # We need to choose two distinct points to form a diagonal.
        for i in range(n):
            p1_list = points[i]
            x1, y1 = p1_list[0], p1_list[1]
            p1_tuple = tuple(p1_list) # Store tuple for set operations later
            
            # Start j from i + 1 to avoid duplicate pairs and considering a point with itself.
            for j in range(i + 1, n):
                p2_list = points[j]
                x2, y2 = p2_list[0], p2_list[1]
                p2_tuple = tuple(p2_list) # Store tuple for set operations later

                # For an axis-aligned rectangle, the x-coordinates and y-coordinates of
                # diagonal points must be different. If they are the same, they form
                # a horizontal or vertical line, not a diagonal.
                if x1 == x2 or y1 == y2:
                    continue
                
                # If p1=(x1, y1) and p2=(x2, y2) are opposite corners, then the other
                # two corners must be p3=(x1, y2) and p4=(x2, y1).
                p3_tuple = (x1, y2)
                p4_tuple = (x2, y1)
                
                # Check if these two potential corners exist in the input points set.
                if p3_tuple in point_set and p4_tuple in point_set:
                    # We have found four points that form a rectangle with axis-parallel edges.
                    
                    # Calculate the area of the current candidate rectangle.
                    # Area = width * height
                    current_area = abs((x2 - x1) * (y2 - y1))
                    
                    # Define the bounding box of this rectangle.
                    # These min/max values will be used to check if other points lie inside or on the border.
                    min_x, max_x = min(x1, x2), max(x1, x2)
                    min_y, max_y = min(y1, y2), max(y1, y2)
                    
                    # Create a set of the four corner points of the current rectangle.
                    # This helps to quickly check if a point being evaluated is one of the rectangle's corners.
                    current_corners_tuples = {p1_tuple, p2_tuple, p3_tuple, p4_tuple}
                    
                    is_valid_rectangle = True
                    # Check every other point from the original 'points' list.
                    # The condition is that NO OTHER point should be inside or on the border.
                    for other_point_list in points:
                        other_point_tuple = tuple(other_point_list)
                        
                        # If this 'other_point_tuple' is one of the four corners of the current rectangle,
                        # it's considered part of the rectangle's definition and is allowed.
                        if other_point_tuple in current_corners_tuples:
                            continue
                        
                        # Check if 'other_point_tuple' lies within or on the boundary of the current rectangle.
                        # If it does, this rectangle is invalid according to the problem statement.
                        if (min_x <= other_point_tuple[0] <= max_x and
                            min_y <= other_point_tuple[1] <= max_y):
                            is_valid_rectangle = False
                            break # Found an invalidating point, no need to check further for this rectangle.
                    
                    # If after checking all other points, the rectangle remains valid,
                    # update the maximum area found so far.
                    if is_valid_rectangle:
                        max_area = max(max_area, current_area)
                        
        return max_area