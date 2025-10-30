class Solution:
    def minimumSubarrayLength(self, nums, k):
        n = len(nums)
        res = float('inf')
        for i in range(n):
            total = 0
            for j in range(i, n):
                total |= nums[j]
                if total >= k:
                    res = min(res, j - i + 1)
                    break
        return res if res != float('inf') else -1