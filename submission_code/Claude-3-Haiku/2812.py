class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # Initialize the maximum achievable number to num
        max_achievable = num
        
        # Iterate t times
        for _ in range(t):
            # Increase the maximum achievable number by 1
            max_achievable += 1
            
            # Decrease the original number by 1
            num -= 1
            
            # Update the maximum achievable number if necessary
            max_achievable = max(max_achievable, num)
        
        return max_achievable