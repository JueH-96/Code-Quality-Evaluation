class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        if n == 0:
            return 0
        
        # Calculate the differences in the original array
        differences = [original[i] - original[i - 1] for i in range(1, n)]
        
        # Initialize the range of possible values for copy[0]
        min_copy = bounds[0][0]
        max_copy = bounds[0][1]
        
        # Iterate through each element and adjust the range of possible values for copy[i]
        for i in range(1, n):
            # Calculate the new range for copy[i] based on copy[i-1] and the difference
            new_min_copy = min_copy + differences[i - 1]
            new_max_copy = max_copy + differences[i - 1]
            
            # Intersect this range with the bounds for copy[i]
            min_copy = max(new_min_copy, bounds[i][0])
            max_copy = min(new_max_copy, bounds[i][1])
            
            # If the range is invalid, return 0
            if min_copy > max_copy:
                return 0
        
        # The number of valid arrays is the size of the final range
        return max_copy - min_copy + 1