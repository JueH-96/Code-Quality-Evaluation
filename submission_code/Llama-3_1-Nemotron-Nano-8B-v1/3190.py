from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        
        # Scenario 1: No swap on the last index
        a1 = nums1[-1]
        b1 = nums2[-1]
        valid1 = True
        swaps1 = 0
        for i in range(n-1):
            current_max = max(nums1[i], nums2[i])
            if current_max > a1 or current_max > b1:
                valid1 = False
                break
        if valid1:
            swaps1 = sum(1 for i in range(n-1) if max(nums1[i], nums2[i]) > a1)
        
        # Scenario 2: Swap the last index
        a2 = nums2[-1]
        b2 = nums1[-1]
        valid2 = True
        swaps2 = 1
        for i in range(n-1):
            if nums2[i] > a2 or nums1[i] > b2:
                valid2 = False
                break
        if valid2:
            swaps2 += sum(1 for i in range(n-1) if nums2[i] > a2 or nums1[i] > b2)
        
        res = []
        if valid1:
            res.append(swaps1)
        if valid2:
            res.append(swaps2)
        
        if not res:
            return -1
        return min(res)