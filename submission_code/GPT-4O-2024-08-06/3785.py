class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        
        # Calculate the differences in the original array
        differences = [original[i] - original[i - 1] for i in range(1, n)]
        
        # Initialize the range for the first element
        min_val, max_val = bounds[0]
        
        # Iterate over each element to determine the valid range for each copy[i]
        for i in range(1, n):
            # Calculate the expected value for copy[i] based on the difference
            expected_min = min_val + differences[i - 1]
            expected_max = max_val + differences[i - 1]
            
            # Update the range for copy[i] based on bounds[i]
            min_val = max(bounds[i][0], expected_min)
            max_val = min(bounds[i][1], expected_max)
            
            # If the range becomes invalid, return 0
            if min_val > max_val:
                return 0
        
        # The number of valid arrays is the size of the range for the last element
        return max_val - min_val + 1