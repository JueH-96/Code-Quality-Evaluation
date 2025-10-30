class Solution:
    def addedInteger(self, nums1: List[int], nums2: List[int]) -> int:
        if not nums1 or not nums2 or len(nums1) != len(nums2):
            return 0
        
        diff = nums2[0] - nums1[0]
        
        for i in range(1, len(nums1)):
            if nums2[i] - nums1[i] != diff:
                return 0
        
        return diff