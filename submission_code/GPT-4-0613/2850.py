class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # sort x and y
        x, y = sorted([x, y])
        # if z is greater than or equal to y, the maximum length is 2 * (x + y)
        if z >= y:
            return 2 * (x + y)
        # if z is less than y, the maximum length is 2 * z + min(x, y - z + 1)
        else:
            return 2 * z + min(x, y - z + 1)