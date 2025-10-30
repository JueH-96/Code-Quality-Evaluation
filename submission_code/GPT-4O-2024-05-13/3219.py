class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(nums[i] - nums[j]) <= limit and nums[i] > nums[j]:
                    nums[i], nums[j] = nums[j], nums[i]
        return nums