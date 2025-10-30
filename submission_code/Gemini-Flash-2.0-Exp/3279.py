class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        ans = 0
        for x in range(1, n + 1):
            if x % 2 == 0:
                ans += (m + 1) // 2
            else:
                ans += m // 2
        return ans