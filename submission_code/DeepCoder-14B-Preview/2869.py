class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        prev_dp1 = 1
        prev_dp2 = 1
        max_len = 1
        for i in range(1, n):
            current_a = nums1[i]
            current_b = nums2[i]
            prev_a = nums1[i-1]
            prev_b = nums2[i-1]
            
            option1 = 1
            if current_a >= prev_a:
                option1 = max(option1, prev_dp1 + 1)
            if current_a >= prev_b:
                option1 = max(option1, prev_dp2 + 1)
            
            option2 = 1
            if current_b >= prev_a:
                option2 = max(option2, prev_dp1 + 1)
            if current_b >= prev_b:
                option2 = max(option2, prev_dp2 + 1)
            
            current_dp1 = option1
            current_dp2 = option2
            
            current_max = max(current_dp1, current_dp2)
            if current_max > max_len:
                max_len = current_max
            
            prev_dp1, prev_dp2 = current_dp1, current_dp2
        
        return max_len