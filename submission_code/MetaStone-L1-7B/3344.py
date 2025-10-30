class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        # Compute u and v for each point
        us = [x + y for x, y in points]
        vs = [x - y for x, y in points]
        
        # Process u values
        u_list = sorted(us)
        u_max = u_list[-1]
        count_u_max = u_list.count(u_max)
        if count_u_max > 1:
            u_max2 = u_max
        else:
            u_max2 = u_list[-2] if len(u_list) >= 2 else u_max
        
        u_min = u_list[0]
        count_u_min = u_list.count(u_min)
        if count_u_min > 1:
            u_min2 = u_min
        else:
            u_min2 = u_list[1] if len(u_list) > 1 else u_min
        
        # Process v values
        v_list = sorted(vs)
        v_max = v_list[-1]
        count_v_max = v_list.count(v_max)
        if count_v_max > 1:
            v_max2 = v_max
        else:
            v_max2 = v_list[-2] if len(v_list) >= 2 else v_max
        
        v_min = v_list[0]
        count_v_min = v_list.count(v_min)
        if count_v_min > 1:
            v_min2 = v_min
        else:
            v_min2 = v_list[1] if len(v_list) > 1 else v_min
        
        min_distance = float('inf')
        
        for x, y in points:
            # Compute new_max_u
            if (x + y) == u_max:
                if count_u_max > 1:
                    new_max_u = u_max
                else:
                    new_max_u = u_max2
            else:
                new_max_u = u_max
            
            # Compute new_min_u
            if (x + y) == u_min:
                if count_u_min > 1:
                    new_min_u = u_min
                else:
                    new_min_u = u_min2
            else:
                new_min_u = u_min
            
            # Compute new_max_v
            if (x - y) == v_max:
                if count_v_max > 1:
                    new_max_v = v_max
                else:
                    new_max_v = v_max2
            else:
                new_max_v = v_max
            
            # Compute new_min_v
            if (x - y) == v_min:
                if count_v_min > 1:
                    new_min_v = v_min
                else:
                    new_min_v = v_min2
            else:
                new_min_v = v_min
            
            current_max = max(new_max_u - new_min_u, new_max_v - new_min_v)
            if current_max < min_distance:
                min_distance = current_max
        
        return min_distance