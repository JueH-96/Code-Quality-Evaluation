class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        prev_dp1 = 1
        prev_dp2 = 1
        max_len = 1
        
        for i in range(1, n):
            curr_dp1 = 1
            curr_dp2 = 1
            
            # Calculate for choosing nums1[i]
            if nums1[i] >= nums1[i-1]:
                curr_dp1 = max(curr_dp1, prev_dp1 + 1)
            if nums1[i] >= nums2[i-1]:
                curr_dp1 = max(curr_dp1, prev_dp2 + 1)
            
            # Calculate for choosing nums2[i]
            if nums2[i] >= nums1[i-1]:
                curr_dp2 = max(curr_dp2, prev_dp1 + 1)
            if nums2[i] >= nums2[i-1]:
                curr_dp2 = max(curr_dp2, prev_dp2 + 1)
            
            # Update the maximum length found so far
            current_max = max(curr_dp1, curr_dp2)
            if current_max > max_len:
                max_len = current_max
            
            # Prepare for next iteration
            prev_dp1, prev_dp2 = curr_dp1, curr_dp2
        
        return max_len