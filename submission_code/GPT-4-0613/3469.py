class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        left, right = 1, 100
        while left <= right:
            mid = (left + right) // 2
            total = mid * (mid + 1) // 2
            if total > red + blue:
                right = mid - 1
            else:
                if total - min(red, blue) <= max(red, blue):
                    left = mid + 1
                else:
                    right = mid - 1
        return right