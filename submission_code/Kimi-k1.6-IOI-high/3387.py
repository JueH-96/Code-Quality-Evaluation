class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        m = n // 2
        cost = 0
        for i in range(n):
            num = nums_sorted[i]
            if i < m:
                cost += max(num - k, 0)
            elif i == m:
                cost += abs(num - k)
            else:
                cost += max(k - num, 0)
        return cost