class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        min_diff = float('inf')
        
        for i in range(1, n):
            max_nums1 = nums[i-1]
            min_nums2 = nums[i]
            min_diff = min(min_diff, abs(max_nums1 - min_nums2))
        
        return min_diff