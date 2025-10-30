class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        plus = [x + y for x, y in points]
        diff = [x - y for x, y in points]
        
        # Compute prefix and suffix max and min for plus
        max_prefix_plus = [0] * n
        min_prefix_plus = [0] * n
        max_prefix_plus[0] = plus[0]
        min_prefix_plus[0] = plus[0]
        for i in range(1, n):
            max_prefix_plus[i] = max(max_prefix_plus[i-1], plus[i])
            min_prefix_plus[i] = min(min_prefix_plus[i-1], plus[i])
        
        max_suffix_plus = [0] * n
        min_suffix_plus = [0] * n
        max_suffix_plus[-1] = plus[-1]
        min_suffix_plus[-1] = plus[-1]
        for i in range(n-2, -1, -1):
            max_suffix_plus[i] = max(plus[i], max_suffix_plus[i+1])
            min_suffix_plus[i] = min(plus[i], min_suffix_plus[i+1])
        
        # Compute prefix and suffix max and min for diff
        max_prefix_diff = [0] * n
        min_prefix_diff = [0] * n
        max_prefix_diff[0] = diff[0]
        min_prefix_diff[0] = diff[0]
        for i in range(1, n):
            max_prefix_diff[i] = max(max_prefix_diff[i-1], diff[i])
            min_prefix_diff[i] = min(min_prefix_diff[i-1], diff[i])
        
        max_suffix_diff = [0] * n
        min_suffix_diff = [0] * n
        max_suffix_diff[-1] = diff[-1]
        min_suffix_diff[-1] = diff[-1]
        for i in range(n-2, -1, -1):
            max_suffix_diff[i] = max(diff[i], max_suffix_diff[i+1])
            min_suffix_diff[i] = min(diff[i], min_suffix_diff[i+1])
        
        min_result = float('inf')
        for i in range(n):
            # Calculate max and min for plus excluding i
            if i == 0:
                max_p = max_suffix_plus[i+1]
                min_p = min_suffix_plus[i+1]
            elif i == n-1:
                max_p = max_prefix_plus[i-1]
                min_p = min_prefix_plus[i-1]
            else:
                max_p = max(max_prefix_plus[i-1], max_suffix_plus[i+1])
                min_p = min(min_prefix_plus[i-1], min_suffix_plus[i+1])
            
            # Calculate max and min for diff excluding i
            if i == 0:
                max_d = max_suffix_diff[i+1]
                min_d = min_suffix_diff[i+1]
            elif i == n-1:
                max_d = max_prefix_diff[i-1]
                min_d = min_prefix_diff[i-1]
            else:
                max_d = max(max_prefix_diff[i-1], max_suffix_diff[i+1])
                min_d = min(min_prefix_diff[i-1], min_suffix_diff[i+1])
            
            current_max = max(max_p - min_p, max_d - min_d)
            if current_max < min_result:
                min_result = current_max
        
        return min_result