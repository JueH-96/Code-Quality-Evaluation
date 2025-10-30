from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0  # according to constraints, this case shouldn't occur
        
        # Case 1: Do not swap the last index
        case1_possible = True
        case1_swaps = 0
        nums1_end = nums1[-1]
        nums2_end = nums2[-1]
        
        for i in range(n-1):
            a, b = nums1[i], nums2[i]
            optionA = (a <= nums1_end) and (b <= nums2_end)
            optionB = (b <= nums1_end) and (a <= nums2_end)
            if not optionA and not optionB:
                case1_possible = False
                break
            if not optionA:
                case1_swaps += 1
        
        # Case 2: Swap the last index
        case2_possible = True
        case2_swaps = 1  # mandatory swap for the last index
        nums1_end_swapped = nums2[-1]
        nums2_end_swapped = nums1[-1]
        
        for i in range(n-1):
            a, b = nums1[i], nums2[i]
            optionA = (a <= nums1_end_swapped) and (b <= nums2_end_swapped)
            optionB = (b <= nums1_end_swapped) and (a <= nums2_end_swapped)
            if not optionA and not optionB:
                case2_possible = False
                break
            if not optionA:
                case2_swaps += 1
        
        # Determine the result
        res = []
        if case1_possible:
            res.append(case1_swaps)
        if case2_possible:
            res.append(case2_swaps)
        
        return min(res) if res else -1