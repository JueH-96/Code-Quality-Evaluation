class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        prev_dp1 = 1
        prev_dp2 = 1
        max_len = 1
        
        for i in range(1, n):
            current_dp1 = 1
            if nums1[i] >= nums1[i-1]:
                current_dp1 = max(current_dp1, prev_dp1 + 1)
            if nums1[i] >= nums2[i-1]:
                current_dp1 = max(current_dp1, prev_dp2 + 1)
            
            current_dp2 = 1
            if nums2[i] >= nums1[i-1]:
                current_dp2 = max(current_dp2, prev_dp1 + 1)
            if nums2[i] >= nums2[i-1]:
                current_dp2 = max(current_dp2, prev_dp2 + 1)
            
            max_len = max(max_len, current_dp1, current_dp2)
            prev_dp1, prev_dp2 = current_dp1, current_dp2
        
        return max_len