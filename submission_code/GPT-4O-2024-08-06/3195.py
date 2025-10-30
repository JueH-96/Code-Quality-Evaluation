class Solution:
    def minimumSteps(self, s: str) -> int:
        # Count the number of black balls (1s)
        black_count = s.count('1')
        
        # Initialize the number of swaps needed
        min_swaps = float('inf')
        
        # Initialize the number of black balls encountered so far
        current_black_count = 0
        
        # Iterate over the string
        for i in range(len(s)):
            if s[i] == '1':
                current_black_count += 1
            
            # Calculate the number of swaps needed if we end the group of black balls here
            swaps = (i + 1 - current_black_count) + (black_count - current_black_count)
            
            # Update the minimum swaps needed
            min_swaps = min(min_swaps, swaps)
        
        return min_swaps