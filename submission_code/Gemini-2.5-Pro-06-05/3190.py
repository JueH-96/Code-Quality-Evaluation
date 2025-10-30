import math
from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)

        def count_prefix_swaps(max1: int, max2: int) -> int:
            """
            Calculates the minimum swaps for indices 0 to n-2 to satisfy
            the given maximums. Returns math.inf if impossible.
            """
            swaps = 0
            for i in range(n - 1):
                n1, n2 = nums1[i], nums2[i]

                # Check if the pair can be valid, either as is or swapped
                keep_ok = (n1 <= max1 and n2 <= max2)
                swap_ok = (n2 <= max1 and n1 <= max2)

                if not keep_ok and not swap_ok:
                    return math.inf
                
                if not keep_ok:
                    # If keeping is not an option, we must swap.
                    swaps += 1
            
            return swaps

        # --- Scenario 1: No swap at the last index ---
        # Target maximums are the original last elements.
        cost1 = count_prefix_swaps(nums1[n - 1], nums2[n - 1])

        # --- Scenario 2: Swap at the last index ---
        # Target maximums are the swapped last elements.
        # This costs 1 operation for the swap at n-1.
        cost2 = count_prefix_swaps(nums2[n - 1], nums1[n - 1])
        if cost2 != math.inf:
            cost2 += 1

        # The result is the minimum cost of the two scenarios.
        min_ops = min(cost1, cost2)

        return min_ops if min_ops != math.inf else -1