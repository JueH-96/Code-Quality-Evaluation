class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        list_a = [x + y for x, y in points]
        list_b = [x - y for x, y in points]
        
        # Precompute for a
        max_a = max(list_a)
        count_max_a = list_a.count(max_a)
        if count_max_a == len(list_a):
            second_max_a = max_a
        else:
            filtered_a = [a for a in list_a if a < max_a]
            second_max_a = max(filtered_a)
        
        min_a = min(list_a)
        count_min_a = list_a.count(min_a)
        if count_min_a == len(list_a):
            second_min_a = min_a
        else:
            filtered_a = [a for a in list_a if a > min_a]
            second_min_a = min(filtered_a)
        
        # Precompute for b
        max_b = max(list_b)
        count_max_b = list_b.count(max_b)
        if count_max_b == len(list_b):
            second_max_b = max_b
        else:
            filtered_b = [b for b in list_b if b < max_b]
            second_max_b = max(filtered_b)
        
        min_b = min(list_b)
        count_min_b = list_b.count(min_b)
        if count_min_b == len(list_b):
            second_min_b = min_b
        else:
            filtered_b = [b for b in list_b if b > min_b]
            second_min_b = min(filtered_b)
        
        min_result = float('inf')
        n = len(points)
        
        for i in range(n):
            a_i = list_a[i]
            b_i = list_b[i]
            
            # Compute new_max_a
            if a_i < max_a:
                new_max_a = max_a
            else:
                if count_max_a > 1:
                    new_max_a = max_a
                else:
                    new_max_a = second_max_a
            
            # Compute new_min_a
            if a_i > min_a:
                new_min_a = min_a
            else:
                if count_min_a > 1:
                    new_min_a = min_a
                else:
                    new_min_a = second_min_a
            
            # Compute new_max_b
            if b_i < max_b:
                new_max_b = max_b
            else:
                if count_max_b > 1:
                    new_max_b = max_b
                else:
                    new_max_b = second_max_b
            
            # Compute new_min_b
            if b_i > min_b:
                new_min_b = min_b
            else:
                if count_min_b > 1:
                    new_min_b = min_b
                else:
                    new_min_b = second_min_b
            
            current_max = max((new_max_a - new_min_a), (new_max_b - new_min_b))
            if current_max < min_result:
                min_result = current_max
        
        return min_result