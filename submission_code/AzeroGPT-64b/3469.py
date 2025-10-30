class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        total = red + blue
        height = 0
        while total > 0:
            height += 1
            total -= height
        if total == 0:
            return height
        else:
            return height - 1