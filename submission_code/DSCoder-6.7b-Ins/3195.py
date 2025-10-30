class Solution:
    def minimumSteps(self, s: str) -> int:
        # Count the number of black and white balls
        black, white = s.count('1'), s.count('0')
        
        # The minimum number of steps is the difference between the number of black balls and white balls
        # We need to move all black balls to the right and all white balls to the left
        # So, we need to move the minimum number of balls from one side to the other
        return abs(black - white)