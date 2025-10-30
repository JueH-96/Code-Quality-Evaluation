class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        # Initialize the range for k based on the first element
        min_k = bounds[0][0] - original[0]
        max_k = bounds[0][1] - original[0]
        
        # Find the intersection of valid k ranges for all elements
        for i in range(1, len(original)):
            u, v = bounds[i]
            
            # Calculate bounds for k based on current element
            lower_k = u - original[i]
            upper_k = v - original[i]
            
            # Update the intersection
            min_k = max(min_k, lower_k)
            max_k = min(max_k, upper_k)
            
            # If intersection becomes empty, no valid arrays exist
            if min_k > max_k:
                return 0
        
        # Number of integers in the range [min_k, max_k] is the answer
        return max_k - min_k + 1