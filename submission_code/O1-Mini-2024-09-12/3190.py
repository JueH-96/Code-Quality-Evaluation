from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)

        # Option 1: Do not swap last index
        max1 = nums1[-1]
        max2 = nums2[-1]

        def compute_swaps(max1, max2):
            swaps = 0
            for i in range(n - 1):
                if nums1[i] <= max1 and nums2[i] <= max2:
                    continue
                elif nums2[i] <= max1 and nums1[i] <= max2:
                    swaps += 1
                else:
                    return -1
            return swaps

        swaps1 = compute_swaps(max1, max2)

        # Option 2: Swap last index
        swapped_max1 = nums2[-1]
        swapped_max2 = nums1[-1]
        swaps2 = compute_swaps(swapped_max1, swapped_max2)
        if swaps2 != -1:
            swaps2 += 1  # Count the swap at the last index

        # Determine the minimum number of swaps
        if swaps1 == -1 and swaps2 == -1:
            return -1
        elif swaps1 == -1:
            return swaps2
        elif swaps2 == -1:
            return swaps1
        else:
            return min(swaps1, swaps2)