class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        total_balls = red + blue
        height = 0
        while total_balls > height:
            height += 1
            total_balls -= height
        if total_balls == height:
            return height
        else:
            return height - 1