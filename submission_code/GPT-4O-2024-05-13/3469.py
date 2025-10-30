class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        height = 0
        while True:
            if height % 2 == 0:
                if red >= height + 1:
                    red -= height + 1
                else:
                    break
            else:
                if blue >= height + 1:
                    blue -= height + 1
                else:
                    break
            height += 1
        return height