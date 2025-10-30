class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        # Compute s and d for all points
        s_values = []
        d_values = []
        for point in points:
            s = point[0] + point[1]
            d = point[0] - point[1]
            s_values.append(s)
            d_values.append(d)
        
        # Find overall max and min for s and d
        overall_max_s = max(s_values)
        overall_min_s = min(s_values)
        overall_max_d = max(d_values)
        overall_min_d = min(d_values)
        
        # Count how many points have the overall max and min for s and d
        count_max_s = s_values.count(overall_max_s)
        count_min_s = s_values.count(overall_min_s)
        count_max_d = d_values.count(overall_max_d)
        count_min_d = d_values.count(overall_min_d)
        
        # Find second_max_s and second_min_s
        if count_max_s > 1:
            second_max_s = overall_max_s
        else:
            second_max_s = max(s for s in s_values if s < overall_max_s)
        
        if count_min_s > 1:
            second_min_s = overall_min_s
        else:
            second_min_s = min(s for s in s_values if s > overall_min_s)
        
        # Find second_max_d and second_min_d
        if count_max_d > 1:
            second_max_d = overall_max_d
        else:
            second_max_d = max(d for d in d_values if d < overall_max_d)
        
        if count_min_d > 1:
            second_min_d = overall_min_d
        else:
            second_min_d = min(d for d in d_values if d > overall_min_d)
        
        # Iterate through each point and compute new_M
        min_new_M = float('inf')
        for i in range(len(points)):
            s = s_values[i]
            d = d_values[i]
            
            # Determine new_max_s and new_min_s
            if s == overall_max_s:
                if count_max_s > 1:
                    new_max_s = overall_max_s
                else:
                    new_max_s = second_max_s
            else:
                new_max_s = overall_max_s
            
            if s == overall_min_s:
                if count_min_s > 1:
                    new_min_s = overall_min_s
                else:
                    new_min_s = second_min_s
            else:
                new_min_s = overall_min_s
            
            # Determine new_max_d and new_min_d
            if d == overall_max_d:
                if count_max_d > 1:
                    new_max_d = overall_max_d
                else:
                    new_max_d = second_max_d
            else:
                new_max_d = overall_max_d
            
            if d == overall_min_d:
                if count_min_d > 1:
                    new_min_d = overall_min_d
                else:
                    new_min_d = second_min_d
            else:
                new_min_d = overall_min_d
            
            # Compute new_S and new_D
            new_S = new_max_s - new_min_s
            new_D = new_max_d - new_min_d
            new_M = max(new_S, new_D)
            
            # Update min_new_M
            if new_M < min_new_M:
                min_new_M = new_M
        
        return min_new_M