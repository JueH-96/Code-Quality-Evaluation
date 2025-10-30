class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        ans = 0
        mod = 10**9 + 7
        for char in s:
            if char == 'z':
                ans = (ans + 2 * t) % mod
            else:
                ans = (ans + 1 + t) % mod
        return ans