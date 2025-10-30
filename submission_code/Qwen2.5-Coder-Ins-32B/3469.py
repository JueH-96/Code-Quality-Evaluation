class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        total_balls = red + blue
        height = 0
        
        while (height * (height + 1)) // 2 <= total_balls:
            if height % 2 == 0:
                if red >= (height // 2 + 1) and blue >= (height // 2):
                    height += 1
                else:
                    break
            else:
                if blue >= ((height // 2) + 1) and red >= (height // 2):
                    height += 1
                else:
                    break
        
        return height - 1