class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        total_balls = red + blue
        height = 0
        while (height + 1) * (height + 2) // 2 <= total_balls:
            height += 1
        if (height + 1) * height // 2 <= max(red, blue):
            return height + 1
        return height