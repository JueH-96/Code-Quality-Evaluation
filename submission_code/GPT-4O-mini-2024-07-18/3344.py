class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        
        # Calculate the max and min for x and y coordinates
        max_x = max(points, key=lambda p: p[0])[0]
        min_x = min(points, key=lambda p: p[0])[0]
        max_y = max(points, key=lambda p: p[1])[1]
        min_y = min(points, key=lambda p: p[1])[1]
        
        # Calculate the maximum distance with all points
        max_distance = (max_x - min_x) + (max_y - min_y)
        
        # To find the minimum possible maximum distance after removing one point
        min_max_distance = float('inf')
        
        for i in range(n):
            x, y = points[i]
            
            # Calculate the new max_x, min_x, max_y, min_y after removing point i
            if x == max_x or x == min_x or y == max_y or y == min_y:
                new_max_x = max_x if x != max_x else max(p[0] for j, p in enumerate(points) if j != i)
                new_min_x = min_x if x != min_x else min(p[0] for j, p in enumerate(points) if j != i)
                new_max_y = max_y if y != max_y else max(p[1] for j, p in enumerate(points) if j != i)
                new_min_y = min_y if y != min_y else min(p[1] for j, p in enumerate(points) if j != i)
                
                # Calculate the new maximum distance
                new_distance = (new_max_x - new_min_x) + (new_max_y - new_min_y)
                min_max_distance = min(min_max_distance, new_distance)
        
        return min_max_distance