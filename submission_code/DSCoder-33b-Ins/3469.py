class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        if red < blue:
            red, blue = blue, red
        if red >= 2 * blue:
            return blue
        else:
            return (red + blue) // 3