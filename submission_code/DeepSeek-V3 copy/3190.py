class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # Determine the maximum values in nums1 and nums2
        max1 = max(nums1)
        max2 = max(nums2)
        
        # Check if the last elements can be the maximums
        if nums1[-1] == max1 and nums2[-1] == max2:
            return 0
        
        # Try swapping the last element
        # Case 1: Swap the last element
        nums1_swap = nums1.copy()
        nums2_swap = nums2.copy()
        nums1_swap[-1], nums2_swap[-1] = nums2_swap[-1], nums1_swap[-1]
        max1_swap = max(nums1_swap)
        max2_swap = max(nums2_swap)
        if nums1_swap[-1] == max1_swap and nums2_swap[-1] == max2_swap:
            return 1
        
        # Case 2: Do not swap the last element, but swap others
        # We need to ensure that the last element of nums1 is the max of nums1
        # and the last element of nums2 is the max of nums2
        # So, for each index, decide whether to swap or not
        # We need to find the minimal number of swaps to make:
        # nums1[-1] is the max of nums1
        # nums2[-1] is the max of nums2
        
        # Initialize the count of swaps
        swap_count = 0
        # Create copies to manipulate
        nums1_copy = nums1.copy()
        nums2_copy = nums2.copy()
        
        # Iterate through each index except the last
        for i in range(n - 1):
            # Check if the current nums1[i] is greater than the last element of nums1
            if nums1_copy[i] > nums1_copy[-1]:
                # Swap to make nums1[-1] the max
                nums1_copy[i], nums2_copy[i] = nums2_copy[i], nums1_copy[i]
                swap_count += 1
            # Check if the current nums2[i] is greater than the last element of nums2
            if nums2_copy[i] > nums2_copy[-1]:
                # Swap to make nums2[-1] the max
                nums1_copy[i], nums2_copy[i] = nums2_copy[i], nums1_copy[i]
                swap_count += 1
        
        # After processing all elements except the last, check if the conditions are met
        if nums1_copy[-1] == max(nums1_copy) and nums2_copy[-1] == max(nums2_copy):
            return swap_count
        
        # If not, try swapping the last element and then process the rest
        # Reset the swap count
        swap_count = 1
        nums1_copy = nums1.copy()
        nums2_copy = nums2.copy()
        nums1_copy[-1], nums2_copy[-1] = nums2_copy[-1], nums1_copy[-1]
        
        for i in range(n - 1):
            if nums1_copy[i] > nums1_copy[-1]:
                nums1_copy[i], nums2_copy[i] = nums2_copy[i], nums1_copy[i]
                swap_count += 1
            if nums2_copy[i] > nums2_copy[-1]:
                nums1_copy[i], nums2_copy[i] = nums2_copy[i], nums1_copy[i]
                swap_count += 1
        
        if nums1_copy[-1] == max(nums1_copy) and nums2_copy[-1] == max(nums2_copy):
            return swap_count
        
        # If none of the above works, return -1
        return -1