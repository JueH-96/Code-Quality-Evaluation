class Solution:
    def minimumSteps(self, s: str) -> int:
        # Count the total number of black balls (1's)
        total_black_balls = s.count('1')
        # Initialize the count of black balls on the right and the minimum steps
        black_balls_on_right, min_steps = 0, total_black_balls
        
        # Iterate over the string from left to right
        for ball in s:
            # If the ball is black, increment the count of black balls on the right
            if ball == '1':
                black_balls_on_right += 1
            # Calculate the number of steps needed if we were to group all black balls
            # to the right at the current position
            else:
                min_steps = min(min_steps, total_black_balls - black_balls_on_right)
        
        return min_steps