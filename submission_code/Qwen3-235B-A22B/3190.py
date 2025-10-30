from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        
        def compute_min_swaps(swap_last: int) -> int:
            if swap_last not in (0, 1):
                return -1
            # Determine x and y based on whether last element is swapped
            if swap_last == 0:
                x = nums1[-1]
                y = nums2[-1]
            else:
                x = nums2[-1]
                y = nums1[-1]
            swap_count = swap_last  # account for swapping last element or not
            valid = True
            
            for i in range(n):
                if i == n - 1:
                    continue  # skip last element processing here
                a = nums1[i]
                b = nums2[i]
                # Check if not swapping is acceptable
                no_swap_ok = (a <= x) and (b <= y)
                # Check if swapping is acceptable
                swap_ok = (b <= x) and (a <= y)
                
                if not no_swap_ok and not swap_ok:
                    valid = False
                    break
                # Choose the option with minimal swaps (prefer no swap if both possible)
                if no_swap_ok and swap_ok:
                    continue
                elif swap_ok:
                    swap_count += 1
                # else no_swap_ok must be True, no increment needed
            
            return swap_count if valid else -1
        
        # Try both scenarios and return the minimal valid result
        res = float('inf')
        for swap_last in [0, 1]:
            current = compute_min_swaps(swap_last)
            if current != -1:
                res = min(res, current)
        
        return res if res != float('inf') else -1