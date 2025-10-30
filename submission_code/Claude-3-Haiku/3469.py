class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        total = red + blue
        height = 0
        
        while total >= height + 1:
            if height % 2 == 0:
                if red >= height + 1:
                    red -= height + 1
                    total -= height + 1
                    height += 1
                else:
                    break
            else:
                if blue >= height + 1:
                    blue -= height + 1
                    total -= height + 1
                    height += 1
                else:
                    break
        
        return height