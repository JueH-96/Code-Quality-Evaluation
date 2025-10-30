class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        ans = 0
        if x == y:
            ans = (x + y + z) * 2
        elif x > y:
            ans = (y + y + 1 + z) * 2
        else:
            ans = (x + x + 1 + z) * 2
        return ans