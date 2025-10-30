class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        u = [x + y for x, y in points]
        v = [x - y for x, y in points]
        
        max1_u, max2_u = -10**18, -10**18
        min1_u, min2_u = 10**18, 10**18
        for val in u:
            if val > max1_u:
                max2_u = max1_u
                max1_u = val
            elif val > max2_u:
                max2_u = val
            if val < min1_u:
                min2_u = min1_u
                min1_u = val
            elif val < min2_u:
                min2_u = val
        
        max1_v, max2_v = -10**18, -10**18
        min1_v, min2_v = 10**18, 10**18
        for val in v:
            if val > max1_v:
                max2_v = max1_v
                max1_v = val
            elif val > max2_v:
                max2_v = val
            if val < min1_v:
                min2_v = min1_v
                min1_v = val
            elif val < min2_v:
                min2_v = val
        
        ans = 10**18
        for i in range(n):
            cur_max_u = max2_u if u[i] == max1_u else max1_u
            cur_min_u = min2_u if u[i] == min1_u else min1_u
            range_u = cur_max_u - cur_min_u
            
            cur_max_v = max2_v if v[i] == max1_v else max1_v
            cur_min_v = min2_v if v[i] == min1_v else min1_v
            range_v = cur_max_v - cur_min_v
            
            current_max = max(range_u, range_v)
            if current_max < ans:
                ans = current_max
        
        return ans