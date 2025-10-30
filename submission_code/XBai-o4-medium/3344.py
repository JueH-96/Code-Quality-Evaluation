from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        if n <= 2:
            return 0  # According to constraints, n >=3, but handle just in case
        
        a_list = [x + y for x, y in points]
        b_list = [x - y for x, y in points]
        
        # Precompute prefix and suffix arrays for a
        prefix_max_a = [0] * n
        prefix_min_a = [0] * n
        suffix_max_a = [0] * n
        suffix_min_a = [0] * n
        
        prefix_max_a[0] = a_list[0]
        prefix_min_a[0] = a_list[0]
        for i in range(1, n):
            prefix_max_a[i] = max(prefix_max_a[i-1], a_list[i])
            prefix_min_a[i] = min(prefix_min_a[i-1], a_list[i])
        
        suffix_max_a[n-1] = a_list[n-1]
        suffix_min_a[n-1] = a_list[n-1]
        for i in range(n-2, -1, -1):
            suffix_max_a[i] = max(a_list[i], suffix_max_a[i+1])
            suffix_min_a[i] = min(a_list[i], suffix_min_a[i+1])
        
        # Precompute prefix and suffix arrays for b
        prefix_max_b = [0] * n
        prefix_min_b = [0] * n
        suffix_max_b = [0] * n
        suffix_min_b = [0] * n
        
        prefix_max_b[0] = b_list[0]
        prefix_min_b[0] = b_list[0]
        for i in range(1, n):
            prefix_max_b[i] = max(prefix_max_b[i-1], b_list[i])
            prefix_min_b[i] = min(prefix_min_b[i-1], b_list[i])
        
        suffix_max_b[n-1] = b_list[n-1]
        suffix_min_b[n-1] = b_list[n-1]
        for i in range(n-2, -1, -1):
            suffix_max_b[i] = max(b_list[i], suffix_max_b[i+1])
            suffix_min_b[i] = min(b_list[i], suffix_min_b[i+1])
        
        min_result = float('inf')
        for i in range(n):
            # Compute for a
            if i == 0:
                max_a = suffix_max_a[1]
                min_a = suffix_min_a[1]
            elif i == n-1:
                max_a = prefix_max_a[n-2]
                min_a = prefix_min_a[n-2]
            else:
                max_a = max(prefix_max_a[i-1], suffix_max_a[i+1])
                min_a = min(prefix_min_a[i-1], suffix_min_a[i+1])
            
            # Compute for b
            if i == 0:
                max_b = suffix_max_b[1]
                min_b = suffix_min_b[1]
            elif i == n-1:
                max_b = prefix_max_b[n-2]
                min_b = prefix_min_b[n-2]
            else:
                max_b = max(prefix_max_b[i-1], suffix_max_b[i+1])
                min_b = min(prefix_min_b[i-1], suffix_min_b[i+1])
            
            current_max = max(max_a - min_a, max_b - min_b)
            if current_max < min_result:
                min_result = current_max
        
        return min_result