class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # Find the maximum values in both arrays
        max1 = max(nums1)
        max2 = max(nums2)
        
        # Check if the last elements are already the maximum
        if nums1[-1] == max1 and nums2[-1] == max2:
            return 0
        
        # Initialize the number of operations needed
        operations = 0
        
        # Check if we can make nums1[n-1] the maximum
        if nums1[-1] != max1:
            # Find if there is a max1 in nums2 that can be swapped to nums1[n-1]
            found = False
            for i in range(n):
                if nums2[i] == max1 and nums1[i] != max1:
                    nums1[i], nums2[i] = nums2[i], nums1[i]
                    operations += 1
                    found = True
                    break
            if not found:
                return -1
        
        # Check if we can make nums2[n-1] the maximum
        if nums2[-1] != max2:
            # Find if there is a max2 in nums1 that can be swapped to nums2[n-1]
            found = False
            for i in range(n):
                if nums1[i] == max2 and nums2[i] != max2:
                    nums1[i], nums2[i] = nums2[i], nums1[i]
                    operations += 1
                    found = True
                    break
            if not found:
                return -1
        
        return operations