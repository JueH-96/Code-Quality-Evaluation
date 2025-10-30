class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        # Transform coordinates: u = x + y, v = x - y
        transformed = []
        for x, y in points:
            transformed.append((x + y, x - y))
        
        # Find points with extreme coordinates
        min_u = min(u for u, v in transformed)
        max_u = max(u for u, v in transformed)
        min_v = min(v for u, v in transformed)
        max_v = max(v for u, v in transformed)
        
        # Find indices of extreme points
        candidates = set()
        for i, (u, v) in enumerate(transformed):
            if u == min_u or u == max_u or v == min_v or v == max_v:
                candidates.add(i)
        
        min_max_dist = float('inf')
        
        # Try removing each candidate point
        for remove_idx in candidates:
            # Calculate max distance without this point
            remaining_u = [u for i, (u, v) in enumerate(transformed) if i != remove_idx]
            remaining_v = [v for i, (u, v) in enumerate(transformed) if i != remove_idx]
            
            if remaining_u:  # Should always be true given constraints
                curr_max_dist = max(max(remaining_u) - min(remaining_u), 
                                  max(remaining_v) - min(remaining_v))
                min_max_dist = min(min_max_dist, curr_max_dist)
        
        return min_max_dist