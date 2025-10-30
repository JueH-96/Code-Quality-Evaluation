class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        m = n // 2
        cost = abs(nums_sorted[m] - k)
        for i in range(m):
            if nums_sorted[i] > k:
                cost += nums_sorted[i] - k
        for i in range(m + 1, n):
            if nums_sorted[i] < k:
                cost += k - nums_sorted[i]
        return cost