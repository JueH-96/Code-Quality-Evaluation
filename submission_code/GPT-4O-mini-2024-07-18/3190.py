class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        max1 = max(nums1)
        max2 = max(nums2)
        
        # Check if the last elements can be the maximum
        if nums1[n - 1] == max1 and nums2[n - 1] == max2:
            return 0
        
        # Count how many swaps are needed
        swaps_needed = 0
        
        # Check if we can make nums1[n-1] the maximum of nums1
        if nums1[n - 1] < max1:
            # We need to find a position where we can swap
            found = False
            for i in range(n):
                if nums2[i] == max1:
                    found = True
                    swaps_needed += 1
                    break
            if not found:
                return -1
        
        # Check if we can make nums2[n-1] the maximum of nums2
        if nums2[n - 1] < max2:
            # We need to find a position where we can swap
            found = False
            for i in range(n):
                if nums1[i] == max2:
                    found = True
                    swaps_needed += 1
                    break
            if not found:
                return -1
        
        return swaps_needed