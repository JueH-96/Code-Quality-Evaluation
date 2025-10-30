class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        max_height = 0
        h = 1
        total_balls = red + blue
        while (h * (h + 1)) // 2 <= total_balls:
            odd_rows = (h + 1) // 2
            even_rows = h // 2
            case1 = (red >= odd_rows * odd_rows) and (blue >= even_rows * (even_rows + 1))
            case2 = (blue >= odd_rows * odd_rows) and (red >= even_rows * (even_rows + 1))
            if case1 or case2:
                max_height = h
            h += 1
        return max_height