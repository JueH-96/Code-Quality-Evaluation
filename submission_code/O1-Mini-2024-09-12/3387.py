class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        m = n // 2
        if nums[m] == k:
            return 0
        elif nums[m] < k:
            total = 0
            for i in range(m, n):
                if nums[i] < k:
                    total += k - nums[i]
            return total
        else:
            total = 0
            for i in range(0, m + 1):
                if nums[i] > k:
                    total += nums[i] - k
            return total