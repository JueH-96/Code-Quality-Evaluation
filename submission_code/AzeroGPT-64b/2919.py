class Solution:
    def minOperations(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        ans = 0
        for i, x in enumerate(usageLimits):
            ans += max(0, x - i)
        return ans

    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        usageLimits.sort()
        ans = pre = 0
        for i, x in enumerate(usageLimits):
            if pre < i + 1:
                ans += 1
                pre += ans
            if pre >= len(usageLimits):
                break
        return ans