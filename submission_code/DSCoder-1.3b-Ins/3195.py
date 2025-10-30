class Solution:
    def minimumSteps(self, s: str) -> int:
        steps = 0
        white_balls = 0
        for ball in s:
            if ball == '1':
                white_balls += 1
            else:
                steps += white_balls
        return steps