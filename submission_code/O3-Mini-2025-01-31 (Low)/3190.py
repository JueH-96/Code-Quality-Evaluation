from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        # We must decide at each index whether or not to swap.
        # After swapping, the final arrays are built as follows:
        #   For each index i, if not swapped then final_i = (nums1[i], nums2[i])
        #   If swapped then final_i = (nums2[i], nums1[i])
        #
        # We want the condition that:
        #   final nums1[n-1] equals the maximum among all final nums1 values,
        #   and final nums2[n-1] equals the maximum among all final nums2 values.
        #
        # Since a swap is allowed at every index including the last one,
        # we have two cases to consider for index n-1:
        # Case 1: Do NOT swap at index n-1.
        #   Then target1 = nums1[n-1] and target2 = nums2[n-1].
        # Case 2: Swap at index n-1.
        #   Then target1 = nums2[n-1] and target2 = nums1[n-1].
        #
        # For the other indices i from 0 to n-2, we have two possible choices:
        # Option A (no swap): final_i = (nums1[i], nums2[i])
        # Option B (swap): final_i = (nums2[i], nums1[i])
        #
        # For each i, to maintain the condition that final[i] does not exceed target in its respective array,
        # we require:
        #   Option A valid if nums1[i] <= target1 and nums2[i] <= target2.
        #   Option B valid if nums2[i] <= target1 and nums1[i] <= target2.
        #
        # We want the minimum number of swaps overall. For each index i, if both options are valid,
        # we choose the one with fewer cost (0 for no swap, 1 for swap).
        # If one of them is valid, we must choose it.
        # If none is valid, that case fails.
        #
        # We'll try both cases for index n-1 and pick the one with a feasible 
        # minimum number of operations.
            
        n = len(nums1)
        res = float('inf')
        
        # Function to compute cost given targets for nums1 and nums2 for positions 0..n-2.
        def compute_cost(target1, target2):
            cost = 0
            for i in range(n-1):
                # check validity and cost for both options
                no_swap_valid = (nums1[i] <= target1 and nums2[i] <= target2)
                swap_valid = (nums2[i] <= target1 and nums1[i] <= target2)
                if not no_swap_valid and not swap_valid:
                    return float('inf')
                # choose minimal cost option (0 if available, else 1)
                if no_swap_valid:
                    # no swap is free, so choose that option.
                    continue
                else:
                    cost += 1
            return cost

        # Case 1: not swapping at index n-1.
        target1 = nums1[n-1]
        target2 = nums2[n-1]
        cost_case1 = compute_cost(target1, target2)
        if cost_case1 != float('inf'):
            # plus 0 for index n-1 because we are not swapping there.
            res = min(res, cost_case1)
        
        # Case 2: swapping at index n-1.
        # After swap, final[n-1] becomes (nums2[n-1], nums1[n-1])
        target1 = nums2[n-1]
        target2 = nums1[n-1]
        cost_case2 = compute_cost(target1, target2)
        if cost_case2 != float('inf'):
            # plus 1 for swapping at n-1.
            res = min(res, cost_case2 + 1)
        
        return res if res != float('inf') else -1

  
# For testing:

if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    nums1 = [1,2,7]
    nums2 = [4,5,3]
    print(sol.minOperations(nums1, nums2))  # Expected output: 1
    
    # Example 2:
    nums1 = [2,3,4,5,9]
    nums2 = [8,8,4,4,4]
    print(sol.minOperations(nums1, nums2))  # Expected output: 2

    # Example 3:
    nums1 = [1,5,4]
    nums2 = [2,5,3]
    print(sol.minOperations(nums1, nums2))  # Expected output: -1