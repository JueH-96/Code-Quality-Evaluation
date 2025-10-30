class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        
        # Calculate the valid range for copy[0]
        # For each i, we have: u_i <= copy[0] + (original[i] - original[0]) <= v_i
        # This gives us: u_i - (original[i] - original[0]) <= copy[0] <= v_i - (original[i] - original[0])
        
        min_start = float('-inf')
        max_start = float('inf')
        
        for i in range(n):
            diff = original[i] - original[0]
            # From bounds[i][0] <= copy[i] <= bounds[i][1]
            # and copy[i] = copy[0] + diff
            # we get bounds[i][0] - diff <= copy[0] <= bounds[i][1] - diff
            min_start = max(min_start, bounds[i][0] - diff)
            max_start = min(max_start, bounds[i][1] - diff)
        
        # If min_start > max_start, no valid array exists
        if min_start > max_start:
            return 0
        
        # Count of valid starting values
        return max_start - min_start + 1