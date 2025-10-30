class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        change = [0] * (n + 1)
        for l, r in queries:
            change[l] += 1
            if r + 1 < n:
                change[r + 1] -= 1
        cnt = [0] * n
        current = 0
        for i in range(n):
            current += change[i]
            cnt[i] = current
        for i in range(n):
            if nums[i] > cnt[i]:
                return False
        return True