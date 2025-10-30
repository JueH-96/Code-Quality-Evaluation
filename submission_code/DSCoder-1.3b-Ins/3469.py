class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        if red == blue:
            return 1
        else:
            return 2