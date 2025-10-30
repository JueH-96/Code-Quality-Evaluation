class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        max1, max2 = max(nums1), max(nums2)
        
        if nums1[-1] == max1 and nums2[-1] == max2:
            return 0
        
        if nums1[-1] == max2 and nums2[-1] == max1:
            return 1
        
        operations = 0
        for i in range(n - 1):
            if nums1[i] > nums1[-1] or nums2[i] > nums2[-1]:
                if nums1[i] <= nums2[-1] and nums2[i] <= nums1[-1]:
                    operations += 1
                else:
                    return -1
        
        return operations