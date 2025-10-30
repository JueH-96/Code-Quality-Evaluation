class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        max_length = 1
        current_length = 1
        
        for i in range(1, n):
            if nums1[i] >= nums1[i - 1] or nums1[i] >= nums2[i - 1]:
                current_length += 1
            else:
                current_length = 1
            
            if nums2[i] >= nums1[i - 1] or nums2[i] >= nums2[i - 1]:
                current_length += 1
            else:
                current_length = 1
            
            max_length = max(max_length, current_length)
        
        return max_length