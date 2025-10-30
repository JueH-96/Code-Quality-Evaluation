class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def required_balls(height, start_color):
            red_needed = 0
            blue_needed = 0
            for i in range(1, height + 1):
                row_color = (start_color + i - 1) % 2 # 0 for red, 1 for blue
                if row_color == 0:
                    red_needed += i
                else:
                    blue_needed += i
            return red_needed, blue_needed

        low = 0
        high = 20 # Maximum possible height is around 20
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            if mid == 0:
                ans = 0
                low = mid + 1
                continue

            red1, blue1 = required_balls(mid, 0)
            red2, blue2 = required_balls(mid, 1)

            possible = False
            if (red1 <= red and blue1 <= blue) or (red2 <= red and blue2 <= blue):
                possible = True

            if possible:
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans