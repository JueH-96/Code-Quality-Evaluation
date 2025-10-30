from typing import List

class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)

        # dp[i][0]: max length of non-decreasing subarray ending at index i, choosing nums1[i]
        # dp[i][1]: max length of non-decreasing subarray ending at index i, choosing nums2[i]
        
        # We use O(1) space by only storing the DP values from the previous index (i-1).
        # Let prev_dp0 be dp[i-1][0] and prev_dp1 be dp[i-1][1].
        # Let curr_dp0 be dp[i][0] and curr_dp1 be dp[i][1].

        # Initialize for the first element (index 0)
        # Any single element forms a non-decreasing subarray of length 1.
        prev_dp0 = 1 # Length ending at 0 using nums1[0]
        prev_dp1 = 1 # Length ending at 0 using nums2[0]
        max_len = 1 # Overall maximum length found so far (at least 1 for n >= 1)

        # Iterate from the second element (index 1) to the last element (n-1)
        for i in range(1, n):
            # Initialize current lengths. At minimum, the element itself forms a subarray of length 1.
            curr_dp0 = 1 
            curr_dp1 = 1

            # Calculate max length ending at i using nums1[i] (curr_dp0)
            # If we chose nums1[i] at current index i, the previous element at i-1 must be <= nums1[i].
            # The previous element could have been nums1[i-1] (represented by prev_dp0)
            # or nums2[i-1] (represented by prev_dp1).
            
            # Option 1: Extend from the subarray ending at i-1 with nums1[i-1]
            if nums1[i-1] <= nums1[i]:
                curr_dp0 = max(curr_dp0, prev_dp0 + 1)
                
            # Option 2: Extend from the subarray ending at i-1 with nums2[i-1]
            if nums2[i-1] <= nums1[i]:
                curr_dp0 = max(curr_dp0, prev_dp1 + 1)

            # Calculate max length ending at i using nums2[i] (curr_dp1)
            # If we chose nums2[i] at current index i, the previous element at i-1 must be <= nums2[i].
            # The previous element could have been nums1[i-1] (represented by prev_dp0)
            # or nums2[i-1] (represented by prev_dp1).
            
            # Option 1: Extend from the subarray ending at i-1 with nums1[i-1]
            if nums1[i-1] <= nums2[i]:
                curr_dp1 = max(curr_dp1, prev_dp0 + 1)
                
            # Option 2: Extend from the subarray ending at i-1 with nums2[i-1]
            if nums2[i-1] <= nums2[i]:
                curr_dp1 = max(curr_dp1, prev_dp1 + 1)

            # The overall maximum length found so far is the max of the previous max length
            # and the lengths ending at the current index i.
            max_len = max(max_len, curr_dp0, curr_dp1)

            # Update the previous DP values for the next iteration (i -> i+1)
            prev_dp0 = curr_dp0
            prev_dp1 = curr_dp1

        return max_len