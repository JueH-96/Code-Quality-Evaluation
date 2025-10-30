class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def can_form_triangle(height, r, b, start_color):
            if start_color == 'red':
                for i in range(1, height+1):
                    if i % 2 == 1:
                        if r < i:
                            return False
                        r -= i
                    else:
                        if b < i:
                            return False
                        b -= i
            else:
                for i in range(1, height+1):
                    if i % 2 == 1:
                        if b < i:
                            return False
                        b -= i
                    else:
                        if r < i:
                            return False
                        r -= i
            return True
        
        max_height = 0
        # Try starting with red
        low = 1
        high = 1000  # Since red and blue are up to 100, the maximum possible height is limited
        while low <= high:
            mid = (low + high) // 2
            if can_form_triangle(mid, red, blue, 'red'):
                max_height = mid
                low = mid + 1
            else:
                high = mid - 1
        # Try starting with blue
        low = 1
        high = 1000
        while low <= high:
            mid = (low + high) // 2
            if can_form_triangle(mid, red, blue, 'blue'):
                max_height = max(max_height, mid)
                low = mid + 1
            else:
                high = mid - 1
        return max_height