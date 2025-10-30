from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        
        def calculate_swaps(swap_last: bool) -> int:
            swap_count = 1 if swap_last else 0
            req1 = nums2[-1] if swap_last else nums1[-1]
            req2 = nums1[-1] if swap_last else nums2[-1]
            
            for i in range(n - 1):
                x, y = nums1[i], nums2[i]
                can_no_swap = x <= req1 and y <= req2
                can_swap = y <= req1 and x <= req2
                
                if not can_no_swap and not can_swap:
                    return -1
                
                if not can_no_swap:
                    swap_count += 1
            return swap_count
        
        res1 = calculate_swaps(False)
        res2 = calculate_swaps(True)
        
        candidates = []
        if res1 != -1:
            candidates.append(res1)
        if res2 != -1:
            candidates.append(res2)
        
        return min(candidates) if candidates else -1