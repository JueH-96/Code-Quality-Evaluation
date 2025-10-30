class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        nums.sort()        
        # We will find the minimum gap between consecutive elements in the sorted array.
        # That gap will be the minimal possible partition value: |max(nums1) - min(nums2)|.
        min_diff = float('inf')
        for i in range(len(nums) - 1):
            min_diff = min(min_diff, nums[i+1] - nums[i])
        return min_diff