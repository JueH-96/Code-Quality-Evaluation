class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        from sys import maxsize
        
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        
        # Calculate the maximum distances for each point removed
        n = len(points)
        if n == 3:
            # If there are only three points, the answer is the maximum distance between any two of the remaining two points
            return min(
                manhattan_distance(points[1], points[2]),
                manhattan_distance(points[0], points[2]),
                manhattan_distance(points[0], points[1])
            )
        
        # Precompute max distances in four directions to optimize the solution
        # max/min x+y and max/min x-y
        max_x_plus_y = -maxsize
        min_x_plus_y = maxsize
        max_x_minus_y = -maxsize
        min_x_minus_y = maxsize
        
        for x, y in points:
            max_x_plus_y = max(max_x_plus_y, x + y)
            min_x_plus_y = min(min_x_plus_y, x + y)
            max_x_minus_y = max(max_x_minus_y, x - y)
            min_x_minus_y = min(min_x_minus_y, x - y)
        
        # Calculate the max distance excluding each point
        min_max_distance = maxsize
        
        for i in range(n):
            x, y = points[i]
            
            # Calculate new max/min excluding the current point
            other_max_x_plus_y = -maxsize
            other_min_x_plus_y = maxsize
            other_max_x_minus_y = -maxsize
            other_min_x_minus_y = maxsize
            
            for j in range(n):
                if i == j:
                    continue
                px, py = points[j]
                other_max_x_plus_y = max(other_max_x_plus_y, px + py)
                other_min_x_plus_y = min(other_min_x_plus_y, px + py)
                other_max_x_minus_y = max(other_max_x_minus_y, px - py)
                other_min_x_minus_y = min(other_min_x_minus_y, px - py)
            
            # Calculate the max distance with the i-th point removed
            max_distance_excluding_i = max(
                other_max_x_plus_y - other_min_x_plus_y,
                other_max_x_minus_y - other_min_x_minus_y
            )
            
            # Update the minimum of these maximum distances
            min_max_distance = min(min_max_distance, max_distance_excluding_i)
        
        return min_max_distance