class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        
        prev_dp1 = 1
        prev_dp2 = 1
        ans = 1
        
        for i in range(1, n):
            curr_dp1 = 1
            curr_dp2 = 1
            
            if nums1[i-1] <= nums1[i]:
                curr_dp1 = max(curr_dp1, prev_dp1 + 1)
            if nums2[i-1] <= nums1[i]:
                curr_dp1 = max(curr_dp1, prev_dp2 + 1)
            
            if nums1[i-1] <= nums2[i]:
                curr_dp2 = max(curr_dp2, prev_dp1 + 1)
            if nums2[i-1] <= nums2[i]:
                curr_dp2 = max(curr_dp2, prev_dp2 + 1)
            
            ans = max(ans, curr_dp1, curr_dp2)
            prev_dp1, prev_dp2 = curr_dp1, curr_dp2
        
        return ans