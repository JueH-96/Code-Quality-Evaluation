class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        nums.sort()
        min_diff = float('inf')
        for i in range(1, len(nums)):
            nums1 = nums[:i]
            nums2 = nums[i:]
            diff = abs(max(nums1) - min(nums2))
            min_diff = min(min_diff, diff)
        return min_diff