from typing import List
import math

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)

        # Helper function to calculate minimum swaps for indices 0 to n-2
        # given target maximums M1 (for final nums1) and M2 (for final nums2).
        # This function assumes that the final nums1[n-1] will be M1
        # and the final nums2[n-1] will be M2.
        # It checks if it's possible to make nums1'[i] <= M1 and nums2'[i] <= M2
        # for all i in [0, n-2] and returns the minimum swaps needed for these indices,
        # or float('inf') if impossible.
        def calculate_swaps(target_max1, target_max2):
            swaps = 0
            for i in range(n - 1):
                v1 = nums1[i]
                v2 = nums2[i]

                # Check if no swap at index i is valid relative to targets target_max1, target_max2
                # i.e., if nums1[i] can be in the final nums1' and nums2[i] in the final nums2'
                # without violating the maximum constraints.
                valid_no_swap = (v1 <= target_max1 and v2 <= target_max2)
                
                # Check if swap at index i is valid relative to targets target_max1, target_max2
                # i.e., if nums2[i] can be in the final nums1' and nums1[i] in the final nums2'
                # without violating the maximum constraints.
                valid_swap = (v2 <= target_max1 and v1 <= target_max2)

                if valid_no_swap and valid_swap:
                    # Both options are valid. Prefer no swap to minimize local cost.
                    pass # swaps += 0
                elif valid_no_swap:
                    # Only no swap is valid. Must not swap.
                    pass # swaps += 0
                elif valid_swap:
                    # Only swap is valid. Must swap.
                    swaps += 1
                else: # not valid_no_swap and not valid_swap
                    # Neither option is valid for index i. Impossible for this target combination.
                    return math.inf

            return swaps

        # Case 1: No swap at index n-1.
        # The target maximums for the final arrays are the original nums1[n-1] and nums2[n-1].
        # The conditions require that the final nums1[n-1] >= all other nums1'[i]
        # and the final nums2[n-1] >= all other nums2'[i].
        # Since we assume no swap at n-1, the final values are nums1[n-1] and nums2[n-1].
        # We need to check if for all i in [0, n-2], we can make nums1'[i] <= nums1[n-1]
        # and nums2'[i] <= nums2[n-1].
        target_max1_case1 = nums1[n-1]
        target_max2_case1 = nums2[n-1]
        
        # Calculate minimum swaps for indices 0 to n-2
        cost0_prefix = calculate_swaps(target_max1_case1, target_max2_case1)
        
        # Total cost for case 1 = cost for indices 0 to n-2 + cost at index n-1 (0 swaps)
        total_cost0 = cost0_prefix

        # Case 2: Swap at index n-1.
        # The target maximums for the final arrays are the original nums2[n-1] and nums1[n-1].
        # We need to check if for all i in [0, n-2], we can make nums1'[i] <= nums2[n-1]
        # and nums2'[i] <= nums1[n-1].
        target_max1_case2 = nums2[n-1]
        target_max2_case2 = nums1[n-1]

        # Calculate minimum swaps for indices 0 to n-2
        cost1_prefix = calculate_swaps(target_max1_case2, target_max2_case2)

        # Total cost for case 2 = cost for indices 0 to n-2 + cost at index n-1 (1 swap)
        total_cost1 = cost1_prefix + 1 if cost1_prefix != math.inf else math.inf

        # The overall minimum is the minimum of the two cases
        min_total_cost = min(total_cost0, total_cost1)

        # If the minimum cost is still infinity, it's impossible
        return min_total_cost if min_total_cost != math.inf else -1