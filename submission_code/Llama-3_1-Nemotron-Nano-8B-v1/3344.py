from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        a_list = [x + y for x, y in points]
        b_list = [x - y for x, y in points]
        
        # Compute for a
        max_a = max(a_list)
        count_max_a = a_list.count(max_a)
        filtered = [a for a in a_list if a < max_a]
        second_max_a = max(filtered) if filtered else max_a
        
        min_a = min(a_list)
        count_min_a = a_list.count(min_a)
        filtered = [a for a in a_list if a > min_a]
        second_min_a = min(filtered) if filtered else min_a
        
        # Compute for b
        max_b = max(b_list)
        count_max_b = b_list.count(max_b)
        filtered = [b for b in b_list if b < max_b]
        second_max_b = max(filtered) if filtered else max_b
        
        min_b = min(b_list)
        count_min_b = b_list.count(min_b)
        filtered = [b for b in b_list if b > min_b]
        second_min_b = min(filtered) if filtered else min_b
        
        min_result = float('inf')
        
        for x, y in points:
            a_val = x + y
            b_val = x - y
            
            # Compute new_max_a
            if a_val == max_a:
                if count_max_a > 1:
                    new_max_a = max_a
                else:
                    new_max_a = second_max_a
            else:
                new_max_a = max_a
            
            # Compute new_min_a
            if a_val == min_a:
                if count_min_a > 1:
                    new_min_a = min_a
                else:
                    new_min_a = second_min_a
            else:
                new_min_a = min_a
            
            # Compute new_max_b
            if b_val == max_b:
                if count_max_b > 1:
                    new_max_b = max_b
                else:
                    new_max_b = second_max_b
            else:
                new_max_b = max_b
            
            # Compute new_min_b
            if b_val == min_b:
                if count_min_b > 1:
                    new_min_b = min_b
                else:
                    new_min_b = second_min_b
            else:
                new_min_b = min_b
            
            current_max = max(new_max_a - new_min_a, new_max_b - new_min_b)
            if current_max < min_result:
                min_result = current_max
        
        return min_result