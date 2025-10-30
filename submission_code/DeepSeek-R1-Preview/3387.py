class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums_sorted = sorted(nums)
        m = len(nums_sorted) // 2
        operations = 0
        for i in range(len(nums_sorted)):
            if i < m:
                if nums_sorted[i] > k:
                    operations += nums_sorted[i] - k
            elif i == m:
                operations += abs(nums_sorted[i] - k)
            else:
                if nums_sorted[i] < k:
                    operations += k - nums_sorted[i]
        return operations