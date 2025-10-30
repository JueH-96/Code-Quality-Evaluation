class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def check(height, r, b, start_color):
            needed_r = 0
            needed_b = 0
            for i in range(1, height + 1):
                if (start_color == 0 and i % 2 == 1) or (start_color == 1 and i % 2 == 0):
                    needed_r += i
                else:
                    needed_b += i
            return needed_r <= r and needed_b <= b

        low = 0
        high = 200 # Maximum possible height is unlikely to exceed 20, but 200 is safe upper bound
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            if mid == 0:
                ans = max(ans, mid)
                low = mid + 1
                continue
            if check(mid, red, blue, 0) or check(mid, red, blue, 1):
                ans = max(ans, mid)
                low = mid + 1
            else:
                high = mid - 1
        return ans