class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0  # Though n >=1 as per constraints
        
        # Initialize lengths for choosing nums1[0] or nums2[0]
        prev1 = 1
        prev2 = 1
        max_len = 1
        
        for i in range(1, n):
            current1 = 1
            current2 = 1
            
            # Update current1 based on nums1[i] and previous choices
            if nums1[i] >= nums1[i-1]:
                current1 = prev1 + 1
            if nums1[i] >= nums2[i-1]:
                current1 = max(current1, prev2 + 1)
            
            # Update current2 based on nums2[i] and previous choices
            if nums2[i] >= nums1[i-1]:
                current2 = max(current2, prev1 + 1)
            if nums2[i] >= nums2[i-1]:
                current2 = max(current2, prev2 + 1)
            
            # Update the maximum length found so far
            max_len = max(max_len, current1, current2)
            
            # Update prev1 and prev2 for the next iteration
            prev1 = current1
            prev2 = current2
        
        return max_len