class Solution:
    def minimizedStringLength(self, s: str) -> int:
        # Create a set to store unique characters in the string
        unique_chars = set(s)
        
        # Initialize the minimized length to the length of the string
        minimized_length = len(s)
        
        # Iterate over each unique character
        for char in unique_chars:
            # Find the indices of the character in the string
            indices = [i for i, c in enumerate(s) if c == char]
            
            # If the character occurs more than once in the string
            if len(indices) > 1:
                # Calculate the minimum difference between consecutive indices
                min_diff = min(indices[i+1] - indices[i] for i in range(len(indices) - 1))
                
                # Update the minimized length
                minimized_length = min(minimized_length, len(s) - len(indices) + min_diff)
        
        # Return the minimized length
        return minimized_length