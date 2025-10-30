from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        if n <= 2:
            return 0  # According to constraints, n >= 3, but handle theoretically
        
        a = [x + y for x, y in points]
        b = [x - y for x, y in points]
        
        # Compute prefix and suffix arrays for a
        prefix_max_a = [0] * n
        prefix_min_a = [0] * n
        prefix_max_a[0] = a[0]
        prefix_min_a[0] = a[0]
        for i in range(1, n):
            prefix_max_a[i] = max(prefix_max_a[i-1], a[i])
            prefix_min_a[i] = min(prefix_min_a[i-1], a[i])
        
        suffix_max_a = [0] * n
        suffix_min_a = [0] * n
        suffix_max_a[-1] = a[-1]
        suffix_min_a[-1] = a[-1]
        for i in range(n-2, -1, -1):
            suffix_max_a[i] = max(suffix_max_a[i+1], a[i])
            suffix_min_a[i] = min(suffix_min_a[i+1], a[i])
        
        # Compute prefix and suffix arrays for b
        prefix_max_b = [0] * n
        prefix_min_b = [0] * n
        prefix_max_b[0] = b[0]
        prefix_min_b[0] = b[0]
        for i in range(1, n):
            prefix_max_b[i] = max(prefix_max_b[i-1], b[i])
            prefix_min_b[i] = min(prefix_min_b[i-1], b[i])
        
        suffix_max_b = [0] * n
        suffix_min_b = [0] * n
        suffix_max_b[-1] = b[-1]
        suffix_min_b[-1] = b[-1]
        for i in range(n-2, -1, -1):
            suffix_max_b[i] = max(suffix_max_b[i+1], b[i])
            suffix_min_b[i] = min(suffix_min_b[i+1], b[i])
        
        min_result = float('inf')
        
        for i in range(n):
            # Calculate new_max_a and new_min_a
            if i == 0:
                new_max_a = suffix_max_a[1]
                new_min_a = suffix_min_a[1]
            elif i == n - 1:
                new_max_a = prefix_max_a[n-2]
                new_min_a = prefix_min_a[n-2]
            else:
                new_max_a = max(prefix_max_a[i-1], suffix_max_a[i+1])
                new_min_a = min(prefix_min_a[i-1], suffix_min_a[i+1])
            
            # Calculate new_max_b and new_min_b
            if i == 0:
                new_max_b = suffix_max_b[1]
                new_min_b = suffix_min_b[1]
            elif i == n - 1:
                new_max_b = prefix_max_b[n-2]
                new_min_b = prefix_min_b[n-2]
            else:
                new_max_b = max(prefix_max_b[i-1], suffix_max_b[i+1])
                new_min_b = min(prefix_min_b[i-1], suffix_min_b[i+1])
            
            a_diff = new_max_a - new_min_a
            b_diff = new_max_b - new_min_b
            current_max = max(a_diff, b_diff)
            
            if current_max < min_result:
                min_result = current_max
        
        return min_result if min_result != float('inf') else 0