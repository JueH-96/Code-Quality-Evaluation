class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        # Initialize min and max possible values for first element
        low = bounds[0][0]
        high = bounds[0][1]
        
        # For each position, calculate the intersection of possible ranges
        for i in range(1, n):
            # Get the difference that needs to be maintained
            diff = original[i] - original[i-1]
            
            # Calculate new possible range for current element based on previous range
            new_low = low + diff
            new_high = high + diff
            
            # Intersect with bounds for current position
            curr_low = bounds[i][0]
            curr_high = bounds[i][1]
            
            # Update the actual possible range
            low = max(new_low, curr_low)
            high = min(new_high, curr_high)
            
            # If range becomes invalid, return 0
            if low > high:
                return 0
            
            # Update range for previous position based on current bounds
            prev_low = low - diff
            prev_high = high - diff
            
            # Intersect with previous bounds
            low = max(prev_low, bounds[i-1][0])
            high = min(prev_high, bounds[i-1][1])
            
            # If range becomes invalid, return 0
            if low > high:
                return 0
        
        # Return number of possible values in the valid range
        return max(0, int(high - low + 1))