class Solution:
    def minimumSteps(self, s: str) -> int:
        # Count the total number of black balls ('1's)
        total_black = s.count('1')
        
        if total_black == 0 or total_black == len(s):
            return 0  # All balls are already grouped
        
        # Find the minimum number of swaps needed
        min_swaps = float('inf')
        current_black = 0  # Number of black balls in the current window
        
        # We use a sliding window of size equal to the number of black balls
        for i in range(len(s)):
            if s[i] == '1':
                current_black += 1
            
            # When the window size exceeds the number of black balls, slide it
            if i >= total_black:
                if s[i - total_black] == '1':
                    current_black -= 1
            
            # Calculate the minimum swaps needed once we have a full window
            if i >= total_black - 1:
                # Minimum swaps needed is the size of the window minus the number of black balls in it
                min_swaps = min(min_swaps, total_black - current_black)
        
        return min_swaps