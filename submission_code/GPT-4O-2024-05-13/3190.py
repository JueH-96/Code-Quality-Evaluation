class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        max1 = max(nums1)
        max2 = max(nums2)
        
        if nums1[-1] == max1 and nums2[-1] == max2:
            return 0
        
        max1_indices = [i for i in range(n) if nums1[i] == max1]
        max2_indices = [i for i in range(n) if nums2[i] == max2]
        
        if not max1_indices and not max2_indices:
            return -1
        
        min_operations = float('inf')
        
        for i in max1_indices:
            if nums2[i] > nums2[-1]:
                min_operations = min(min_operations, 1)
        
        for i in max2_indices:
            if nums1[i] > nums1[-1]:
                min_operations = min(min_operations, 1)
        
        if min_operations == float('inf'):
            return -1
        
        return min_operations