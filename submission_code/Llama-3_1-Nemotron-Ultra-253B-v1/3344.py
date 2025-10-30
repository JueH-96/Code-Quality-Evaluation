from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        s = [x + y for x, y in points]
        d = [x - y for x, y in points]
        
        # Precompute values for s
        max_s = -float('inf')
        second_max_s = -float('inf')
        count_max_s = 0
        min_s = float('inf')
        second_min_s = float('inf')
        count_min_s = 0
        
        for x in s:
            # Update max_s and second_max_s
            if x > max_s:
                second_max_s = max_s
                max_s = x
                count_max_s = 1
            elif x == max_s:
                count_max_s += 1
            elif x > second_max_s:
                second_max_s = x
            
            # Update min_s and second_min_s
            if x < min_s:
                second_min_s = min_s
                min_s = x
                count_min_s = 1
            elif x == min_s:
                count_min_s += 1
            elif x < second_min_s:
                second_min_s = x
        
        if second_max_s == -float('inf'):
            second_max_s = max_s
        if second_min_s == float('inf'):
            second_min_s = min_s
        
        # Precompute values for d
        max_d = -float('inf')
        second_max_d = -float('inf')
        count_max_d = 0
        min_d = float('inf')
        second_min_d = float('inf')
        count_min_d = 0
        
        for x in d:
            # Update max_d and second_max_d
            if x > max_d:
                second_max_d = max_d
                max_d = x
                count_max_d = 1
            elif x == max_d:
                count_max_d += 1
            elif x > second_max_d:
                second_max_d = x
            
            # Update min_d and second_min_d
            if x < min_d:
                second_min_d = min_d
                min_d = x
                count_min_d = 1
            elif x == min_d:
                count_min_d += 1
            elif x < second_min_d:
                second_min_d = x
        
        if second_max_d == -float('inf'):
            second_max_d = max_d
        if second_min_d == float('inf'):
            second_min_d = min_d
        
        min_result = float('inf')
        for i in range(len(points)):
            # Calculate new_max_s and new_min_s for s
            if s[i] == max_s:
                new_max_s = max_s if count_max_s > 1 else second_max_s
            else:
                new_max_s = max_s
            
            if s[i] == min_s:
                new_min_s = min_s if count_min_s > 1 else second_min_s
            else:
                new_min_s = min_s
            
            new_s_diff = new_max_s - new_min_s
            
            # Calculate new_max_d and new_min_d for d
            if d[i] == max_d:
                new_max_d = max_d if count_max_d > 1 else second_max_d
            else:
                new_max_d = max_d
            
            if d[i] == min_d:
                new_min_d = min_d if count_min_d > 1 else second_min_d
            else:
                new_min_d = min_d
            
            new_d_diff = new_max_d - new_min_d
            
            current_max = max(new_s_diff, new_d_diff)
            if current_max < min_result:
                min_result = current_max
        
        return min_result