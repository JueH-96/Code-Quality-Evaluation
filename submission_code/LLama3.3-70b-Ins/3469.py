class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def check(height):
            r, b = red, blue
            for i in range(1, height + 1):
                if i % 2 == 1:
                    if r < i:
                        return False
                    r -= i
                else:
                    if b < i:
                        return False
                    b -= i
            return True

        def check2(height):
            r, b = red, blue
            for i in range(1, height + 1):
                if i % 2 == 1:
                    if b < i:
                        return False
                    b -= i
                else:
                    if r < i:
                        return False
                    r -= i
            return True

        left, right = 1, min(red, blue) * 2
        while left < right:
            mid = (left + right + 1) // 2
            if check(mid) or check2(mid):
                left = mid
            else:
                right = mid - 1
        return left