class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        mod = 10**9 + 7
        
        # Calculate the differences between consecutive elements in original
        diffs = [original[i] - original[i - 1] for i in range(1, n)]
        
        # Initialize the number of valid arrays
        count = 1
        
        # Initialize the previous lower and upper bounds for the first element
        prev_lower = bounds[0][0]
        prev_upper = bounds[0][1]
        
        for i in range(1, n):
            # Calculate the new lower and upper bounds based on the previous bounds and the difference
            new_lower = max(bounds[i][0], prev_lower + diffs[i - 1])
            new_upper = min(bounds[i][1], prev_upper + diffs[i - 1])
            
            # If the new bounds are valid
            if new_lower <= new_upper:
                count = count * (new_upper - new_lower + 1) % mod
            
            # Update the previous bounds for the next iteration
            prev_lower = bounds[i][0]
            prev_upper = bounds[i][1]
        
        return count