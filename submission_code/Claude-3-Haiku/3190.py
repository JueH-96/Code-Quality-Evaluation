class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # Find the maximum values in nums1 and nums2
        max_nums1 = max(nums1)
        max_nums2 = max(nums2)
        
        # Check if both conditions are already satisfied
        if nums1[-1] == max_nums1 and nums2[-1] == max_nums2:
            return 0
        
        # Check if it's impossible to satisfy both conditions
        if max_nums1 < nums2[-1] or max_nums2 < nums1[-1]:
            return -1
        
        # Perform the swapping operations
        operations = 0
        for i in range(n):
            if nums1[i] != max_nums1 and nums2[i] != max_nums2:
                if nums1[i] < max_nums2 - nums1[i]:
                    nums1[i], nums2[i] = nums2[i], nums1[i]
                    operations += 1
                elif nums2[i] < max_nums1 - nums2[i]:
                    nums1[i], nums2[i] = nums2[i], nums1[i]
                    operations += 1
        
        # Check if both conditions are now satisfied
        if nums1[-1] == max_nums1 and nums2[-1] == max_nums2:
            return operations
        else:
            return -1