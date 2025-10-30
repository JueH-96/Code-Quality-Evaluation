class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        nums.sort()
        n = len(nums)
        
        min_diff = float('inf')
        
        # Try all possible partitions
        # nums1: [0...i], nums2: [i+1...n-1]
        for i in range(n - 1):
            # max of nums1 is nums[i]
            # min of nums2 is nums[i+1]
            diff = nums[i + 1] - nums[i]
            min_diff = min(min_diff, diff)
        
        return min_diff