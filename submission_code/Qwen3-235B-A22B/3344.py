class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        if n <= 2:
            return 0  # As per problem constraints, n >=3
        
        u_list = [x + y for x, y in points]
        v_list = [x - y for x, y in points]
        
        # Compute u stats: max_u, min_u, count_max_u, count_min_u, second_max_u, second_min_u
        max_u = max(u_list)
        min_u = min(u_list)
        count_max_u = u_list.count(max_u)
        count_min_u = u_list.count(min_u)
        
        # Compute second_max_u and second_min_u
        second_max_u = None
        for u in u_list:
            if u < max_u:
                if second_max_u is None or u > second_max_u:
                    second_max_u = u
        if second_max_u is None:
            second_max_u = max_u
        
        second_min_u = None
        for u in u_list:
            if u > min_u:
                if second_min_u is None or u < second_min_u:
                    second_min_u = u
        if second_min_u is None:
            second_min_u = min_u
        
        # Compute v stats: max_v, min_v, count_max_v, count_min_v, second_max_v, second_min_v
        max_v = max(v_list)
        min_v = min(v_list)
        count_max_v = v_list.count(max_v)
        count_min_v = v_list.count(min_v)
        
        # Compute second_max_v and second_min_v
        second_max_v = None
        for v in v_list:
            if v < max_v:
                if second_max_v is None or v > second_max_v:
                    second_max_v = v
        if second_max_v is None:
            second_max_v = max_v
        
        second_min_v = None
        for v in v_list:
            if v > min_v:
                if second_min_v is None or v < second_min_v:
                    second_min_v = v
        if second_min_v is None:
            second_min_v = min_v
        
        # Collect all candidate points
        candidates = []
        for i in range(n):
            x, y = points[i]
            u = x + y
            v = x - y
            if u == max_u or u == min_u or v == max_v or v == min_v:
                candidates.append(i)
        
        min_result = float('inf')
        for idx in candidates:
            x, y = points[idx]
            u = x + y
            v_val = x - y
            
            # Compute new_max_u and new_min_u
            if u == max_u:
                if count_max_u == 1:
                    new_max_u = second_max_u
                else:
                    new_max_u = max_u
            else:
                new_max_u = max_u
            
            if u == min_u:
                if count_min_u == 1:
                    new_min_u = second_min_u
                else:
                    new_min_u = min_u
            else:
                new_min_u = min_u
            
            new_spread_u = new_max_u - new_min_u
            
            # Compute new_max_v and new_min_v
            if v_val == max_v:
                if count_max_v == 1:
                    new_max_v = second_max_v
                else:
                    new_max_v = max_v
            else:
                new_max_v = max_v
            
            if v_val == min_v:
                if count_min_v == 1:
                    new_min_v = second_min_v
                else:
                    new_min_v = min_v
            else:
                new_min_v = min_v
            
            new_spread_v = new_max_v - new_min_v
            
            current_max = max(new_spread_u, new_spread_v)
            if current_max < min_result:
                min_result = current_max
        
        return min_result