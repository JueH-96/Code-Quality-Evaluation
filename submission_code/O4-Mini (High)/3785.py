from typing import List

class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        # The copy array is determined by copy[i] = original[i] + d for some integer d.
        # For each index i, we need u_i <= original[i] + d <= v_i
        # => u_i - original[i] <= d <= v_i - original[i].
        # So we take the intersection of all those intervals for d.
        
        n = len(original)
        # Initialize with the constraints from index 0
        low = bounds[0][0] - original[0]
        high = bounds[0][1] - original[0]
        
        # Intersect with constraints from the rest of the indices
        for i in range(1, n):
            u, v = bounds[i]
            o = original[i]
            low = max(low, u - o)
            high = min(high, v - o)
            if low > high:
                return 0
        
        # The number of integer d in [low, high] is high - low + 1 (or 0 if invalid)
        return max(0, high - low + 1)