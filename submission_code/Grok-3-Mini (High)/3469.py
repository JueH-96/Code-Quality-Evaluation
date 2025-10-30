class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        low = 1
        high = 200  # Sufficiently large upper bound
        result = 0
        while low <= high:
            mid = (low + high) // 2
            ceil_mid_2 = (mid + 1) // 2
            floor_mid_2 = mid // 2
            if (red >= ceil_mid_2 ** 2 and blue >= floor_mid_2 * (floor_mid_2 + 1)) or \
               (blue >= ceil_mid_2 ** 2 and red >= floor_mid_2 * (floor_mid_2 + 1)):
                result = mid
                low = mid + 1
            else:
                high = mid - 1
        return result