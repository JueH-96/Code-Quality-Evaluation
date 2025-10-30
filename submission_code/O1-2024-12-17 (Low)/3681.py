class Solution:
    def maxRectangleArea(self, points: List[List[int]]) -> int:
        # Convert the list of points to a set for faster membership checks
        point_set = set(tuple(p) for p in points)
        
        # Extract all unique x and y coordinates
        xs = sorted(set(p[0] for p in points))
        ys = sorted(set(p[1] for p in points))
        
        max_area = -1
        
        # We'll try all pairs of x and y values to form potential rectangles
        for i in range(len(xs)):
            for j in range(i+1, len(xs)):
                x1, x2 = xs[i], xs[j]
                for k in range(len(ys)):
                    for l in range(k+1, len(ys)):
                        y1, y2 = ys[k], ys[l]
                        
                        # Check if all four corners exist in the set
                        corners = [(x1,y1), (x1,y2), (x2,y1), (x2,y2)]
                        if all(c in point_set for c in corners):
                            # Check that no other point lies within or on the border
                            # except these four corners
                            valid = True
                            for px, py in points:
                                # If (px, py) is not one of the corners and lies
                                # in or on the rectangle, it's invalid
                                if (px, py) not in corners:
                                    if x1 <= px <= x2 and y1 <= py <= y2:
                                        valid = False
                                        break
                            
                            if valid:
                                area = (x2 - x1) * (y2 - y1)
                                max_area = max(max_area, area)
        
        return max_area