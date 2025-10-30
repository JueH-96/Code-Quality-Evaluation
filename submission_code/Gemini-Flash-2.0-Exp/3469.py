class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def check(height, red, blue):
            total = height * (height + 1) // 2
            if red < 0 or blue < 0:
                return False
            if red + blue < total:
                return False
            
            for i in range(2):
                red_needed = 0
                for j in range(1, height + 1):
                    if (j % 2 == 1 and i == 0) or (j % 2 == 0 and i == 1):
                        red_needed += j
                
                if red_needed <= red and total - red_needed <= blue:
                    return True
            return False

        low = 0
        high = red + blue
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            if check(mid, red, blue):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans