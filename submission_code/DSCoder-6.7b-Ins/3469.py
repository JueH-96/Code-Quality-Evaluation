class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        # The maximum height of the triangle is the minimum of red and blue
        # because we can only use one color at a time.
        # The triangle will be a pyramid shape, so the height is the number of rows.
        # The number of rows is the minimum of red and blue because we can only use one color at a time.
        return min(red, blue)