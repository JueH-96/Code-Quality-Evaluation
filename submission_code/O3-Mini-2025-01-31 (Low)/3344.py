from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        
        # Precompute global min and max for x and y as well as second min and max.
        # For x:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        
        # Helper function to get min, second min and count of min.
        def get_min_info(arr):
            min_val = min(arr)
            count = arr.count(min_val)
            second_min = min(val for val in arr if val != min_val) if count == 1 or len(arr) > count else min_val
            return min_val, second_min, count
        
        def get_max_info(arr):
            max_val = max(arr)
            count = arr.count(max_val)
            second_max = max(val for val in arr if val != max_val) if count == 1 or len(arr) > count else max_val
            return max_val, second_max, count
        
        min_x, second_min_x, count_min_x = get_min_info(xs)
        max_x, second_max_x, count_max_x = get_max_info(xs)
        min_y, second_min_y, count_min_y = get_min_info(ys)
        max_y, second_max_y, count_max_y = get_max_info(ys)
        
        # The maximum Manhattan distance among points is given by (max_x - min_x) + (max_y - min_y).
        # When removing a point, if that point is an extreme (min or max) and unique,
        # then the extreme will shift to second_min or second_max accordingly.
        
        ans = float('inf')
        
        for (x, y) in points:
            # Determine new min_x after removal of this point
            if x == min_x and count_min_x == 1:
                curr_min_x = second_min_x
            else:
                curr_min_x = min_x
                
            if x == max_x and count_max_x == 1:
                curr_max_x = second_max_x
            else:
                curr_max_x = max_x
                
            if y == min_y and count_min_y == 1:
                curr_min_y = second_min_y
            else:
                curr_min_y = min_y
                
            if y == max_y and count_max_y == 1:
                curr_max_y = second_max_y
            else:
                curr_max_y = max_y
                
            curr_range = (curr_max_x - curr_min_x) + (curr_max_y - curr_min_y)
            ans = min(ans, curr_range)
        
        return ans