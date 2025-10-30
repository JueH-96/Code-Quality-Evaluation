class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def can_form_triangle(height, red, blue):
            total_red = 0
            total_blue = 0
            for i in range(1, height + 1):
                if i % 2 == 1:
                    total_red += i
                else:
                    total_blue += i
            if total_red <= red and total_blue <= blue:
                return True
            total_red = 0
            total_blue = 0
            for i in range(1, height + 1):
                if i % 2 == 1:
                    total_blue += i
                else:
                    total_red += i
            if total_red <= red and total_blue <= blue:
                return True
            return False
        
        low = 1
        high = 1000
        max_height = 0
        while low <= high:
            mid = (low + high) // 2
            if can_form_triangle(mid, red, blue):
                max_height = mid
                low = mid + 1
            else:
                high = mid - 1
        return max_height