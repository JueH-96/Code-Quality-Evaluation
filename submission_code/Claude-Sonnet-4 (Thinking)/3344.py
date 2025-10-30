class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        # Transform coordinates: (x, y) -> (x+y, x-y)
        # Manhattan distance becomes Chebyshev distance
        transformed = [(x + y, x - y) for x, y in points]
        
        u_coords = [u for u, v in transformed]
        v_coords = [v for u, v in transformed]
        
        max_u, min_u = max(u_coords), min(u_coords)
        max_v, min_v = max(v_coords), min(v_coords)
        
        from collections import Counter
        u_count = Counter(u_coords)
        v_count = Counter(v_coords)
        
        # Find second largest/smallest values
        u_sorted = sorted(set(u_coords))
        v_sorted = sorted(set(v_coords))
        
        second_max_u = u_sorted[-2] if len(u_sorted) > 1 else max_u
        second_min_u = u_sorted[1] if len(u_sorted) > 1 else min_u
        second_max_v = v_sorted[-2] if len(v_sorted) > 1 else max_v
        second_min_v = v_sorted[1] if len(v_sorted) > 1 else min_v
        
        result = float('inf')
        
        for i, (u, v) in enumerate(transformed):
            # Only consider removing extreme points
            if not (u in {max_u, min_u} or v in {max_v, min_v}):
                continue
            
            # Calculate new extremes after removing point i
            new_max_u = second_max_u if u == max_u and u_count[max_u] == 1 else max_u
            new_min_u = second_min_u if u == min_u and u_count[min_u] == 1 else min_u
            new_max_v = second_max_v if v == max_v and v_count[max_v] == 1 else max_v
            new_min_v = second_min_v if v == min_v and v_count[min_v] == 1 else min_v
            
            max_dist = max(new_max_u - new_min_u, new_max_v - new_min_v)
            result = min(result, max_dist)
        
        return result