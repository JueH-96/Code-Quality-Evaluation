class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # Calculate the maximum values in both arrays
        max1 = max(nums1)
        max2 = max(nums2)
        
        # If the last elements are already the maximums
        if nums1[n-1] == max1 and nums2[n-1] == max2:
            return 0
        
        # If swapping the last elements makes both conditions true
        if nums1[n-1] == max2 and nums2[n-1] == max1:
            return 1
        
        # We need to check if it's possible to make nums1[n-1] and nums2[n-1] the maximums
        # by swapping some elements.
        # We will check if the maximum of nums1 can be moved to the last position
        # and the maximum of nums2 can be moved to the last position with minimum swaps.
        
        # Find the indices of the maximum values
        max1_indices = [i for i in range(n) if nums1[i] == max1]
        max2_indices = [i for i in range(n) if nums2[i] == max2]
        
        # Check if we can swap to make both conditions true
        # We need nums1[n-1] to be max1 and nums2[n-1] to be max2
        min_swaps = float('inf')
        
        # Check all combinations of indices where max1 and max2 are located
        for i in max1_indices:
            for j in max2_indices:
                if i == j:
                    # If max1 and max2 are at the same index, we need one swap if it's not the last index
                    if i != n-1:
                        min_swaps = min(min_swaps, 1)
                else:
                    # If they are at different indices, check if one of them is already at the last index
                    swaps = 0
                    if i != n-1:
                        swaps += 1
                    if j != n-1:
                        swaps += 1
                    min_swaps = min(min_swaps, swaps)
        
        return min_swaps if min_swaps != float('inf') else -1