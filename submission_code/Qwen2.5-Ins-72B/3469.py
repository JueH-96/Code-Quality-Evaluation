class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def can_form_triangle(height, red, blue):
            total_red = (height // 2) * ((height // 2) + 1)
            total_blue = (height // 2) * ((height // 2) + 1)
            if height % 2 == 1:
                total_red += (height // 2) + 1
            return total_red <= red and total_blue <= blue
        
        left, right = 1, 100
        max_height = 0
        while left <= right:
            mid = (left + right) // 2
            if can_form_triangle(mid, red, blue) or can_form_triangle(mid, blue, red):
                max_height = mid
                left = mid + 1
            else:
                right = mid - 1
        return max_height