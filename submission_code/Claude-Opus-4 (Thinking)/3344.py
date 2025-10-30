class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        
        # Transform points to use Chebyshev distance
        transformed = [(x + y, x - y) for x, y in points]
        
        # Extract a and b values
        a_values = [p[0] for p in transformed]
        b_values = [p[1] for p in transformed]
        
        # Find extreme values
        min_a = min(a_values)
        max_a = max(a_values)
        min_b = min(b_values)
        max_b = max(b_values)
        
        # Find critical points (those with extreme values)
        critical_indices = set()
        for i in range(n):
            if a_values[i] == min_a or a_values[i] == max_a:
                critical_indices.add(i)
            if b_values[i] == min_b or b_values[i] == max_b:
                critical_indices.add(i)
        
        # Try removing each critical point
        min_max_dist = float('inf')
        for remove_idx in critical_indices:
            # Compute max distance without this point
            remaining_a = [a_values[i] for i in range(n) if i != remove_idx]
            remaining_b = [b_values[i] for i in range(n) if i != remove_idx]
            
            range_a = max(remaining_a) - min(remaining_a)
            range_b = max(remaining_b) - min(remaining_b)
            max_dist = max(range_a, range_b)
            
            min_max_dist = min(min_max_dist, max_dist)
        
        return min_max_dist