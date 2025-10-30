class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def maxHeight(first, second):
            h = 0
            while True:
                h += 1
                if h % 2 == 1:  # Odd row - use first color
                    if first >= h:
                        first -= h
                    else:
                        return h - 1
                else:  # Even row - use second color
                    if second >= h:
                        second -= h
                    else:
                        return h - 1
        
        # Try both configurations: red first, then blue first
        return max(maxHeight(red, blue), maxHeight(blue, red))