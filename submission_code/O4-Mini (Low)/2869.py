from typing import List

class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        # f1: length ending at i choosing nums1[i]
        # f2: length ending at i choosing nums2[i]
        f1 = f2 = 1
        ans = 1
        
        for i in range(1, n):
            new_f1 = 1
            new_f2 = 1
            
            # if we pick nums1[i], check transitions from previous picks
            if nums1[i] >= nums1[i - 1]:
                new_f1 = max(new_f1, f1 + 1)
            if nums1[i] >= nums2[i - 1]:
                new_f1 = max(new_f1, f2 + 1)
            
            # if we pick nums2[i], check transitions from previous picks
            if nums2[i] >= nums1[i - 1]:
                new_f2 = max(new_f2, f1 + 1)
            if nums2[i] >= nums2[i - 1]:
                new_f2 = max(new_f2, f2 + 1)
            
            f1, f2 = new_f1, new_f2
            ans = max(ans, f1, f2)
        
        return ans