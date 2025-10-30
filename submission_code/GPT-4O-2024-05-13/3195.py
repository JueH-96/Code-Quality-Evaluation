class Solution:
    def minimumSteps(self, s: str) -> int:
        # Count the number of black balls (1s)
        count_black = s.count('1')
        
        # Initialize the minimum steps to a large number
        min_steps = float('inf')
        
        # Initialize the number of swaps needed to move all black balls to the right
        swaps = 0
        
        # Iterate through the string
        for i in range(len(s)):
            if s[i] == '1':
                # Calculate the number of swaps needed if we consider this position as the start of the black balls
                swaps += i - (count_black - 1)
                count_black -= 1
                min_steps = min(min_steps, swaps)
        
        return min_steps