class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # dp1[i] will store the length of the longest non-decreasing subarray ending at index i
        # if we choose nums1[i] for nums3[i]
        dp1 = [0] * n
        
        # dp2[i] will store the length of the longest non-decreasing subarray ending at index i
        # if we choose nums2[i] for nums3[i]
        dp2 = [0] * n
        
        # Initialize the first element
        dp1[0] = 1
        dp2[0] = 1
        
        # Iterate through each index and calculate the dp values
        for i in range(1, n):
            # If we choose nums1[i] for nums3[i]
            if nums1[i] >= nums1[i - 1]:
                dp1[i] = dp1[i - 1] + 1
            else:
                dp1[i] = 1
            
            if nums1[i] >= nums2[i - 1]:
                dp1[i] = max(dp1[i], dp2[i - 1] + 1)
            
            # If we choose nums2[i] for nums3[i]
            if nums2[i] >= nums2[i - 1]:
                dp2[i] = dp2[i - 1] + 1
            else:
                dp2[i] = 1
            
            if nums2[i] >= nums1[i - 1]:
                dp2[i] = max(dp2[i], dp1[i - 1] + 1)
        
        # The result is the maximum value in both dp arrays
        return max(max(dp1), max(dp2))