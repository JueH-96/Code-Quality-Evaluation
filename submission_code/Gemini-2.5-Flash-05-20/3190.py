import math

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)

        # Helper function to calculate minimum operations for elements at indices 0 to n-2,
        # given the fixed maximum targets for nums1 and nums2.
        # max1_target: The required maximum value for the final nums1 array.
        # max2_target: The required maximum value for the final nums2 array.
        def calculate_ops(max1_target: int, max2_target: int) -> int:
            ops = 0
            # Iterate through elements from index 0 up to n-2.
            # The elements at index n-1 are the targets themselves.
            for i in range(n - 1):
                curr1 = nums1[i]
                curr2 = nums2[i]

                # Option 1: Keep values as they are (no swap at index i).
                # Check if current values satisfy the conditions.
                can_no_swap = (curr1 <= max1_target) and (curr2 <= max2_target)
                
                # Option 2: Swap values at index i.
                # Check if swapped values satisfy the conditions.
                can_swap = (curr2 <= max1_target) and (curr1 <= max2_target)

                if can_no_swap and can_swap:
                    # Both options are valid. We choose not to swap to minimize operations.
                    # ops remains unchanged (cost 0).
                    pass 
                elif can_no_swap:
                    # Only not swapping works.
                    # ops remains unchanged (cost 0).
                    pass
                elif can_swap:
                    # Only swapping works. We must perform a swap.
                    ops += 1
                else:
                    # Neither keeping nor swapping the current pair satisfies the conditions.
                    # This path is impossible. Return infinity to signify this.
                    return math.inf
            return ops

        min_total_ops = math.inf

        # Scenario 1: Do not swap nums1[n-1] and nums2[n-1].
        # The target maximums are their original values at n-1.
        target_max1_case1 = nums1[n - 1]
        target_max2_case1 = nums2[n - 1]
        ops_for_case1 = calculate_ops(target_max1_case1, target_max2_case1)
        min_total_ops = min(min_total_ops, ops_for_case1)

        # Scenario 2: Swap nums1[n-1] and nums2[n-1].
        # The target maximums become nums2[n-1] for nums1 and nums1[n-1] for nums2.
        # This initial swap at index n-1 costs 1 operation.
        target_max1_case2 = nums2[n - 1]
        target_max2_case2 = nums1[n - 1]
        ops_for_case2 = calculate_ops(target_max1_case2, target_max2_case2)
        
        # If ops_for_case2 is not infinity, add 1 for the swap at n-1.
        # Otherwise, adding 1 to infinity is still infinity, but explicit check is safer.
        if ops_for_case2 != math.inf:
            min_total_ops = min(min_total_ops, ops_for_case2 + 1)

        # If min_total_ops is still infinity, it means no valid sequence of operations was found.
        return min_total_ops if min_total_ops != math.inf else -1