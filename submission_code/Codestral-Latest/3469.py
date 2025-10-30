class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        height = 0
        while red > 0 and blue > 0:
            if height % 2 == 0:
                if red > 0:
                    red -= 1
                else:
                    break
            else:
                if blue > 0:
                    blue -= 1
                else:
                    break
            height += 1

        if red > 0 and height % 2 == 0:
            height += 1
        elif blue > 0 and height % 2 != 0:
            height += 1

        return height