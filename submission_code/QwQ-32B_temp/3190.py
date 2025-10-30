from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0  # According to constraints, n >= 1
        
        original_last1 = nums1[-1]
        original_last2 = nums2[-1]
        
        def compute_swaps(target1: int, target2: int) -> int:
            swaps = 0
            possible = True
            for i in range(n - 1):  # Check all except the last element
                a = nums1[i]
                b = nums2[i]
                
                # Check option 1: do not swap
                opt1 = (a <= target1) and (b <= target2)
                # Check option 2: swap
                opt2 = (b <= target1) and (a <= target2)
                
                if not opt1 and not opt2:
                    possible = False
                    break
                
                # Choose the option with minimal swaps (prefer not swapping if possible)
                if opt1 and opt2:
                    # Choose not to swap (0 swaps)
                    continue
                elif opt1:
                    continue  # 0 swaps
                else:  # must swap
                    swaps += 1
            return swaps if possible else -1
        
        # Case 1: do not swap the last elements
        case1 = compute_swaps(original_last1, original_last2)
        
        # Case 2: swap the last elements, so targets are swapped
        case2 = compute_swaps(original_last2, original_last1)
        if case2 != -1:
            case2 += 1  # add the swap of the last element
        
        # Determine the result
        result = float('inf')
        if case1 != -1:
            result = min(result, case1)
        if case2 != -1:
            result = min(result, case2)
        
        return result if result != float('inf') else -1