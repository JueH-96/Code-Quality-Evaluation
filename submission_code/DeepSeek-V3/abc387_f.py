class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        if n == 0:
            return 0
        
        # The copy array must satisfy copy[i] = copy[0] + (original[i] - original[0])
        # So for each i, bounds[i][0] <= copy[0] + (original[i] - original[0]) <= bounds[i][1]
        # Which translates to:
        # copy[0] >= bounds[i][0] - (original[i] - original[0])
        # copy[0] <= bounds[i][1] - (original[i] - original[0])
        
        lower = -float('inf')
        upper = float('inf')
        original_0 = original[0]
        
        for i in range(n):
            delta = original[i] - original_0
            current_lower = bounds[i][0] - delta
            current_upper = bounds[i][1] - delta
            
            lower = max(lower, current_lower)
            upper = min(upper, current_upper)
            
            if lower > upper:
                return 0
        
        return upper - lower + 1 if upper >= lower else 0