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
        # Case 1: Do not swap the last element
        # Check if the last element of nums1 is the maximum of nums1 and last of nums2 is the maximum of nums2
        if nums1[-1] == max1 and nums2[-1] == max2:
            # No need to swap, but we need to ensure that all other elements are less than or equal to their respective last elements
            # Check for all i < n-1, nums1[i] <= nums1[-1] and nums2[i] <= nums2[-1]
            valid = True
            for i in range(n-1):
                if nums1[i] > nums1[-1] or nums2[i] > nums2[-1]:
                    valid = False
                    break
            if valid:
                return 0
        
        # Case 2: Swap the last element
        # Swap nums1[-1] and nums2[-1]
        nums1[-1], nums2[-1] = nums2[-1], nums1[-1]
        # Now, check if the new last elements are the maximums
        new_max1 = max(nums1)
        new_max2 = max(nums2)
        if nums1[-1] == new_max1 and nums2[-1] == new_max2:
            # Now, check if all other elements are less than or equal to their respective last elements
            valid = True
            for i in range(n-1):
                if nums1[i] > nums1[-1] or nums2[i] > nums2[-1]:
                    valid = False
                    break
            if valid:
                return 1
        # Swap back to original
        nums1[-1], nums2[-1] = nums2[-1], nums1[-1]
        
        # Now, try swapping other elements
        # We need to find the minimal number of swaps such that:
        # nums1[-1] is the maximum of nums1 and nums2[-1] is the maximum of nums2
        # and all other elements are less than or equal to their respective last elements
        
        # To achieve this, we can consider two scenarios:
        # Scenario 1: nums1[-1] is the maximum of nums1 and nums2[-1] is the maximum of nums2
        # Scenario 2: nums1[-1] is the maximum of nums2 and nums2[-1] is the maximum of nums1
        
        # Let's try both scenarios and find the minimal number of swaps
        
        # Scenario 1: nums1[-1] is the maximum of nums1 and nums2[-1] is the maximum of nums2
        # We need to ensure that for all i < n-1, nums1[i] <= nums1[-1] and nums2[i] <= nums2[-1]
        # We can swap elements at positions where nums1[i] > nums1[-1] or nums2[i] > nums2[-1]
        # Count the number of such swaps
        swap_count1 = 0
        valid1 = True
        for i in range(n):
            if nums1[i] > nums1[-1] or nums2[i] > nums2[-1]:
                # Swap nums1[i] and nums2[i]
                nums1[i], nums2[i] = nums2[i], nums1[i]
                swap_count1 += 1
                # After swapping, check if the conditions are still met
                if nums1[i] > nums1[-1] or nums2[i] > nums2[-1]:
                    valid1 = False
                    break
        if valid1:
            # Now, check if nums1[-1] is the maximum of nums1 and nums2[-1] is the maximum of nums2
            if nums1[-1] == max(nums1) and nums2[-1] == max(nums2):
                # All conditions are satisfied
                pass
            else:
                valid1 = False
        
        # Scenario 2: nums1[-1] is the maximum of nums2 and nums2[-1] is the maximum of nums1
        # Swap the last elements first
        nums1[-1], nums2[-1] = nums2[-1], nums1[-1]
        swap_count2 = 1
        valid2 = True
        for i in range(n):
            if nums1[i] > nums1[-1] or nums2[i] > nums2[-1]:
                # Swap nums1[i] and nums2[i]
                nums1[i], nums2[i] = nums2[i], nums1[i]
                swap_count2 += 1
                # After swapping, check if the conditions are still met
                if nums1[i] > nums1[-1] or nums2[i] > nums2[-1]:
                    valid2 = False
                    break
        if valid2:
            # Now, check if nums1[-1] is the maximum of nums1 and nums2[-1] is the maximum of nums2
            if nums1[-1] == max(nums1) and nums2[-1] == max(nums2):
                # All conditions are satisfied
                pass
            else:
                valid2 = False
        
        # Swap back to original
        nums1[-1], nums2[-1] = nums2[-1], nums1[-1]
        
        # Determine the minimal swap count
        if valid1 and valid2:
            return min(swap_count1, swap_count2)
        elif valid1:
            return swap_count1
        elif valid2:
            return swap_count2
        else:
            return -1