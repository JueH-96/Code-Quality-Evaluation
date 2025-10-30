from typing import List

class Solution:
  def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
    n = len(nums1)
    
    # Constraints: 1 <= n <= 10^5. So n is at least 1.
    # If n is 1, the loop `for i in range(1, n)` (i.e. range(1,1)) will not execute.
    # In this scenario, `max_overall_length` (initialized to 1) will be returned. This is correct,
    # as a single element array has a longest non-decreasing subarray of length 1.
    
    # dp_prev_from_nums1: Stores the length of the LNDS ending at index i-1,
    #                     assuming nums3[i-1] was chosen as nums1[i-1].
    # dp_prev_from_nums2: Stores the length of the LNDS ending at index i-1,
    #                     assuming nums3[i-1] was chosen as nums2[i-1].
    
    # Initialize DP values for index i = 0. These will serve as the "previous" state
    # for the first iteration of the loop (when i=1).
    # If nums3[0] is nums1[0], LNDS is [nums1[0]], length 1.
    # If nums3[0] is nums2[0], LNDS is [nums2[0]], length 1.
    dp_prev_from_nums1 = 1
    dp_prev_from_nums2 = 1
    
    # max_overall_length stores the maximum LNDS length found so far across all possible
    # subarrays and choices for nums3 elements.
    # Since n >= 1, any single element forms an LNDS of length 1, so min possible max_overall_length is 1.
    max_overall_length = 1
    
    # Iterate from the second element (index 1) up to the last element (index n-1).
    for i in range(1, n):
        # Calculate LNDS ending at index i if nums3[i] is chosen as nums1[i].
        # Initialize to 1, as [nums1[i]] itself is an LNDS of length 1.
        dp_curr_from_nums1 = 1
        
        # Option 1: nums3[i-1] was nums1[i-1]. Check if nums1[i-1] <= nums1[i].
        if nums1[i-1] <= nums1[i]:
            dp_curr_from_nums1 = max(dp_curr_from_nums1, dp_prev_from_nums1 + 1)
        
        # Option 2: nums3[i-1] was nums2[i-1]. Check if nums2[i-1] <= nums1[i].
        if nums2[i-1] <= nums1[i]:
            dp_curr_from_nums1 = max(dp_curr_from_nums1, dp_prev_from_nums2 + 1)
            
        # Calculate LNDS ending at index i if nums3[i] is chosen as nums2[i].
        # Initialize to 1, as [nums2[i]] itself is an LNDS of length 1.
        dp_curr_from_nums2 = 1
        
        # Option 1: nums3[i-1] was nums1[i-1]. Check if nums1[i-1] <= nums2[i].
        if nums1[i-1] <= nums2[i]:
            dp_curr_from_nums2 = max(dp_curr_from_nums2, dp_prev_from_nums1 + 1)
            
        # Option 2: nums3[i-1] was nums2[i-1]. Check if nums2[i-1] <= nums2[i].
        if nums2[i-1] <= nums2[i]:
            dp_curr_from_nums2 = max(dp_curr_from_nums2, dp_prev_from_nums2 + 1)
            
        # Update the overall maximum length found so far.
        max_overall_length = max(max_overall_length, dp_curr_from_nums1, dp_curr_from_nums2)
        
        # The DP values calculated for index 'i' become the "previous" DP values for index 'i+1'.
        dp_prev_from_nums1 = dp_curr_from_nums1
        dp_prev_from_nums2 = dp_curr_from_nums2
            
    return max_overall_length