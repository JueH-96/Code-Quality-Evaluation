from typing import List

class Solution:
  """
  Finds the maximum length of a non-decreasing subarray that can be formed 
  by choosing elements from nums1 or nums2 at each index.
  Uses dynamic programming with O(1) space complexity.
  """
  def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
    """
    Calculates the maximum length of a non-decreasing subarray.

    Args:
      nums1: The first input array.
      nums2: The second input array. Both nums1 and nums2 have the same length n.

    Returns:
      An integer representing the maximum length of the longest non-decreasing subarray.
    """
    n = len(nums1)
    # Constraints state n >= 1. If n=0 was possible, we'd return 0.
    # Since n >= 1, the minimum possible length is 1.

    # Initialize DP states. These variables store the length of the 
    # longest non-decreasing subarray ending at the *previous* index (i-1).
    # dp1: length ending at i-1 using nums1[i-1]
    # dp2: length ending at i-1 using nums2[i-1]
    # Initialize for the base case, index 0: the length is 1 regardless of choice.
    dp1 = 1 
    dp2 = 1
    
    # max_len stores the overall maximum length found so far across all possible subarrays.
    # Initialize with the base case length 1, as any single element forms a non-decreasing subarray of length 1.
    max_len = 1

    # Iterate from the second element (index 1) up to the end of the arrays (index n-1).
    for i in range(1, n):
        # Calculate potential lengths ending at the current index i.
        
        # Calculate length ending at i if we choose nums1[i] for nums3[i].
        # Check if we can extend the subarray ending at i-1 with nums1[i-1].
        # If nums1[i] >= nums1[i-1], the potential length is dp1 + 1. 
        # Otherwise, this path breaks the non-decreasing property, so we start a new subarray of length 1.
        c11 = dp1 + 1 if nums1[i] >= nums1[i-1] else 1
        
        # Check if we can extend the subarray ending at i-1 with nums2[i-1].
        # If nums1[i] >= nums2[i-1], the potential length is dp2 + 1.
        # Otherwise, this path breaks the non-decreasing property, starting a new subarray of length 1.
        c12 = dp2 + 1 if nums1[i] >= nums2[i-1] else 1
        
        # The maximum length ending at index i using nums1[i] is the maximum of the two possibilities.
        current_dp1 = max(c11, c12)

        # Calculate length ending at i if we choose nums2[i] for nums3[i].
        # Check if we can extend the subarray ending at i-1 with nums1[i-1].
        # If nums2[i] >= nums1[i-1], potential length is dp1 + 1. Else, length is 1.
        c21 = dp1 + 1 if nums2[i] >= nums1[i-1] else 1
        
        # Check if we can extend the subarray ending at i-1 with nums2[i-1].
        # If nums2[i] >= nums2[i-1], potential length is dp2 + 1. Else, length is 1.
        c22 = dp2 + 1 if nums2[i] >= nums2[i-1] else 1
        
        # The maximum length ending at index i using nums2[i] is the maximum of these two possibilities.
        current_dp2 = max(c21, c22)

        # Update the overall maximum length found so far.
        # The maximum length could end at the current index i, either using nums1[i] or nums2[i].
        max_len = max(max_len, current_dp1, current_dp2)

        # Update the DP states for the next iteration (i+1).
        # The calculated lengths for the current index i become the 'previous' lengths (dp1, dp2) for the next step.
        dp1 = current_dp1
        dp2 = current_dp2
            
    # After iterating through all indices, max_len holds the maximum length found among all possible non-decreasing subarrays.
    return max_len