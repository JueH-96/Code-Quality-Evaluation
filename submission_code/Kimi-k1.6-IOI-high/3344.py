class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        if n == 1:
            return 0  # Though constraints say n >=3, this is a safeguard
        
        # Initialize prefix arrays
        prefix_max_plus = [0] * n
        prefix_min_plus = [0] * n
        prefix_max_minus = [0] * n
        prefix_min_minus = [0] * n
        
        prefix_max_plus[0] = points[0][0] + points[0][1]
        prefix_min_plus[0] = points[0][0] + points[0][1]
        prefix_max_minus[0] = points[0][0] - points[0][1]
        prefix_min_minus[0] = points[0][0] - points[0][1]
        
        for i in range(1, n):
            plus = points[i][0] + points[i][1]
            minus = points[i][0] - points[i][1]
            prefix_max_plus[i] = max(prefix_max_plus[i-1], plus)
            prefix_min_plus[i] = min(prefix_min_plus[i-1], plus)
            prefix_max_minus[i] = max(prefix_max_minus[i-1], minus)
            prefix_min_minus[i] = min(prefix_min_minus[i-1], minus)
        
        # Initialize suffix arrays
        suffix_max_plus = [0] * n
        suffix_min_plus = [0] * n
        suffix_max_minus = [0] * n
        suffix_min_minus = [0] * n
        
        suffix_max_plus[-1] = points[-1][0] + points[-1][1]
        suffix_min_plus[-1] = points[-1][0] + points[-1][1]
        suffix_max_minus[-1] = points[-1][0] - points[-1][1]
        suffix_min_minus[-1] = points[-1][0] - points[-1][1]
        
        for i in range(n-2, -1, -1):
            plus = points[i][0] + points[i][1]
            minus = points[i][0] - points[i][1]
            suffix_max_plus[i] = max(suffix_max_plus[i+1], plus)
            suffix_min_plus[i] = min(suffix_min_plus[i+1], plus)
            suffix_max_minus[i] = max(suffix_max_minus[i+1], minus)
            suffix_min_minus[i] = min(suffix_min_minus[i+1], minus)
        
        min_result = float('inf')
        
        for i in range(n):
            # Calculate new_max_plus
            left_max_plus = prefix_max_plus[i-1] if i > 0 else -float('inf')
            right_max_plus = suffix_max_plus[i+1] if i < n-1 else -float('inf')
            new_max_plus = max(left_max_plus, right_max_plus)
            
            # Calculate new_min_plus
            left_min_plus = prefix_min_plus[i-1] if i > 0 else float('inf')
            right_min_plus = suffix_min_plus[i+1] if i < n-1 else float('inf')
            new_min_plus = min(left_min_plus, right_min_plus)
            
            # Calculate new_max_minus
            left_max_minus = prefix_max_minus[i-1] if i > 0 else -float('inf')
            right_max_minus = suffix_max_minus[i+1] if i < n-1 else -float('inf')
            new_max_minus = max(left_max_minus, right_max_minus)
            
            # Calculate new_min_minus
            left_min_minus = prefix_min_minus[i-1] if i > 0 else float('inf')
            right_min_minus = suffix_min_minus[i+1] if i < n-1 else float('inf')
            new_min_minus = min(left_min_minus, right_min_minus)
            
            # Compute current max distance
            dist1 = new_max_plus - new_min_plus
            dist2 = new_max_minus - new_min_minus
            current_max = max(dist1, dist2)
            
            if current_max < min_result:
                min_result = current_max
        
        return min_result