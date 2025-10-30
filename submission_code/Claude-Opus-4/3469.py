class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def getMaxHeight(first, second):
            height = 0
            row = 1
            
            while True:
                if row % 2 == 1:  # Odd rows use first color
                    if first >= row:
                        first -= row
                        height += 1
                    else:
                        break
                else:  # Even rows use second color
                    if second >= row:
                        second -= row
                        height += 1
                    else:
                        break
                row += 1
            
            return height
        
        # Try starting with red, then blue
        height1 = getMaxHeight(red, blue)
        # Try starting with blue, then red
        height2 = getMaxHeight(blue, red)
        
        return max(height1, height2)