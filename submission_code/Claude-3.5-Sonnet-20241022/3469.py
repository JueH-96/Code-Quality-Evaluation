class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def canBuildTriangle(h, color1, color2):
            # Try building triangle of height h starting with color1
            balls1 = 0  # balls needed of color1
            balls2 = 0  # balls needed of color2
            
            for i in range(1, h + 1):
                if i % 2 == 1:
                    balls1 += i
                else:
                    balls2 += i
                    
            return (balls1 <= color1 and balls2 <= color2) or (balls1 <= color2 and balls2 <= color1)
        
        # Binary search for maximum height
        left, right = 1, 100
        result = 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if canBuildTriangle(mid, red, blue):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
                
        return result