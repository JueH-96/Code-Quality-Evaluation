from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        
        # Check scenario 1: do not swap the last index
        a1 = nums1[-1]
        b1 = nums2[-1]
        count1 = 0
        possible1 = True
        for i in range(n - 1):
            x, y = nums1[i], nums2[i]
            if x <= a1 and y <= b1:
                continue
            elif y <= a1 and x <= b1:
                count1 += 1
            else:
                possible1 = False
                break
        
        # Check scenario 2: swap the last index
        a2 = nums2[-1]
        b2 = nums1[-1]
        count2 = 0
        possible2 = True
        for i in range(n - 1):
            x, y = nums1[i], nums2[i]
            if x <= a2 and y <= b2:
                continue
            elif y <= a2 and x <= b2:
                count2 += 1
            else:
                possible2 = False
                break
        
        # Compute possible results
        res = []
        if possible1:
            res.append(count1)
        if possible2:
            res.append(1 + count2)
        
        return min(res) if res else -1