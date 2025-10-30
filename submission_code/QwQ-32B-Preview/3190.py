from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 1:
            if nums1[0] == max(nums1[0], nums2[0]):
                return 0
            else:
                return 1
        
        # Scenario 1: Do not swap the last elements
        nums1_last1 = nums1[-1]
        nums2_last1 = nums2[-1]
        dp1 = [[float('inf')] * 2 for _ in range(n)]
        dp1[0][0] = 0  # No swap at index 0
        dp1[0][1] = 1  # Swap at index 0
        
        for i in range(1, n):
            # Option 1: Do not swap at index i
            if (nums1[i] <= nums1_last1 and nums2[i] <= nums2_last1):
                dp1[i][0] = min(dp1[i][0], dp1[i-1][0])  # Continue without swapping
                dp1[i][0] = min(dp1[i][0], dp1[i-1][1])  # Swap previous but not this
            # Option 2: Swap at index i
            if (nums2[i] <= nums1_last1 and nums1[i] <= nums2_last1):
                dp1[i][1] = min(dp1[i][1], dp1[i-1][0] + 1)  # Swap this, not previous
                dp1[i][1] = min(dp1[i][1], dp1[i-1][1] + 1)  # Swap previous and this
        
        # Check if scenario 1 is feasible
        feasible1 = True
        max_nums1_1 = max(nums1[:-1] + [nums1_last1])
        max_nums2_1 = max(nums2[:-1] + [nums2_last1])
        if max_nums1_1 != nums1_last1 or max_nums2_1 != nums2_last1:
            feasible1 = False
        
        # Scenario 2: Swap the last elements
        nums1_last2 = nums2[-1]
        nums2_last2 = nums1[-1]
        dp2 = [[float('inf')] * 2 for _ in range(n)]
        dp2[0][0] = 0  # No swap at index 0
        dp2[0][1] = 1  # Swap at index 0
        
        for i in range(1, n):
            # Option 1: Do not swap at index i
            if (nums1[i] <= nums1_last2 and nums2[i] <= nums2_last2):
                dp2[i][0] = min(dp2[i][0], dp2[i-1][0])  # Continue without swapping
                dp2[i][0] = min(dp2[i][0], dp2[i-1][1])  # Swap previous but not this
            # Option 2: Swap at index i
            if (nums2[i] <= nums1_last2 and nums1[i] <= nums2_last2):
                dp2[i][1] = min(dp2[i][1], dp2[i-1][0] + 1)  # Swap this, not previous
                dp2[i][1] = min(dp2[i][1], dp2[i-1][1] + 1)  # Swap previous and this
        
        # Check if scenario 2 is feasible
        feasible2 = True
        nums1_swapped = nums1[:-1] if n > 1 else []
        nums2_swapped = nums2[:-1] if n > 1 else []
        max_nums1_2 = max(nums1_swapped + [nums1_last2])
        max_nums2_2 = max(nums2_swapped + [nums2_last2])
        if max_nums1_2 != nums1_last2 or max_nums2_2 != nums2_last2:
            feasible2 = False
        
        # Determine the result
        res1 = min(dp1[n-1][0], dp1[n-1][1]) if feasible1 else float('inf')
        res2 = min(dp2[n-1][0], dp2[n-1][1]) + 1 if feasible2 else float('inf')
        
        final_res = min(res1, res2)
        return final_res if final_res != float('inf') else -1