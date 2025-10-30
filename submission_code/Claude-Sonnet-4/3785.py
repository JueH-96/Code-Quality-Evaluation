class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        
        # Find the valid range for copy[0]
        min_val = float('-inf')
        max_val = float('inf')
        
        for i in range(n):
            # copy[i] = copy[0] + (original[i] - original[0])
            # bounds[i][0] <= copy[i] <= bounds[i][1]
            # bounds[i][0] <= copy[0] + (original[i] - original[0]) <= bounds[i][1]
            # bounds[i][0] - (original[i] - original[0]) <= copy[0] <= bounds[i][1] - (original[i] - original[0])
            
            diff = original[i] - original[0]
            left_bound = bounds[i][0] - diff
            right_bound = bounds[i][1] - diff
            
            min_val = max(min_val, left_bound)
            max_val = min(max_val, right_bound)
        
        # If min_val > max_val, there's no valid range
        if min_val > max_val:
            return 0
        
        # Count integers in the range [min_val, max_val]
        return max_val - min_val + 1