class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        ans = k - 1
        for x in range(1, k + 1):
            if x > ans:
                break
            n = (k + x - 1) // x
            cost = x + n - 2
            if cost < ans:
                ans = cost
        return ans