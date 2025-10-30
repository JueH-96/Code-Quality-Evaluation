class Solution:
    def findValueOfPartition(self, nums: List[int]) -> int:
        # Sort the array to easily identify how to split
        nums.sort()
        
        # The minimum value of the partition will be determined by the minimal
        # difference between consecutive elements in the sorted array.
        # This is because if we split between two consecutive elements,
        # those elements become (max(nums1), min(nums2)).
        
        min_difference = float('inf')
        
        for i in range(1, len(nums)):
            diff = nums[i] - nums[i - 1]
            if diff < min_difference:
                min_difference = diff
        
        return min_difference