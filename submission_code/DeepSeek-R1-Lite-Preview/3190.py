from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        
        # Define a function to compute minimum swaps for given target_max1 and target_max2
        def compute_min_swaps(target_max1, target_max2, swap_n_1):
            total_swaps = 0
            for i in range(n - 1):
                # Option 1: Don't swap
                option1_possible = (nums1[i] <= target_max1) and (nums2[i] <= target_max2)
                # Option 2: Swap
                option2_possible = (nums2[i] <= target_max1) and (nums1[i] <= target_max2)
                
                if option1_possible and option2_possible:
                    # Choose not to swap, cost 0
                    continue
                elif option1_possible:
                    # Choose not to swap, cost 0
                    continue
                elif option2_possible:
                    # Choose to swap, cost 1
                    total_swaps += 1
                else:
                    # Neither option is possible
                    return float('inf')
            
            # Handle the last index based on swap_n_1
            if swap_n_1:
                if (nums2[n - 1] <= target_max1) and (nums1[n - 1] <= target_max2):
                    total_swaps += 1
                else:
                    return float('inf')
            else:
                if (nums1[n - 1] <= target_max1) and (nums2[n - 1] <= target_max2):
                    # No additional swaps needed
                    pass
                else:
                    return float('inf')
            
            return total_swaps
        
        # Option A: Do not swap at n-1
        target_max1_A = nums1[-1]
        target_max2_A = nums2[-1]
        swaps_A = compute_min_swaps(target_max1_A, target_max2_A, swap_n_1=False)
        
        # Option B: Swap at n-1
        target_max1_B = nums2[-1]
        target_max2_B = nums1[-1]
        swaps_B = compute_min_swaps(target_max1_B, target_max2_B, swap_n_1=True)
        
        # Determine the minimum swaps required
        min_swaps = min(swaps_A, swaps_B)
        if min_swaps == float('inf'):
            return -1
        else:
            return min_swaps