class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        idx = n // 2
        total_operations = 0
        if nums[idx] == k:
            return 0
        elif nums[idx] < k:
            for i in range(idx, n):
                if nums[i] < k:
                    total_operations += k - nums[i]
        else:
            for i in range(0, idx + 1):
                if nums[i] > k:
                    total_operations += nums[i] - k
        return total_operations