class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        height = 0
        while red >= height + 1 and blue >= height + 1:
            red -= height + 1
            blue -= height + 1
            height += 1
        return height