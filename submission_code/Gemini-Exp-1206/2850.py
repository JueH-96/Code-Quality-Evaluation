class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        ans = z * 2
        m = min(x, y)
        ans += m * 4
        if x > m or y > m:
            ans += 2
        return ans