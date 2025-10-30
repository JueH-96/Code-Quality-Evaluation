class Solution:
    def findMaximumNumber(self, k: int, x: int) -> int:
        count = [0] * 60
        for i in range(1, 60):
            count[i] = count[i - 1] + (1 if i % x == 0 else 0)

        ans = 0
        for i in range(59, -1, -1):
            if k >= count[i]:
                k -= count[i]
                ans |= (1 << i)
            elif i % x == 0:
                break

        return ans