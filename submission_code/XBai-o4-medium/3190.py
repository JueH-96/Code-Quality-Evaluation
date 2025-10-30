from typing import List

class Solution:
    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        if n == 0:
            return 0
        
        def calculate(target1: int, target2: int) -> int:
            swaps = 0
            for i in range(n):
                a = nums1[i]
                b = nums2[i]
                if a <= target1 and b <= target2:
                    continue
                elif b <= target1 and a <= target2:
                    swaps += 1
                else:
                    return float('inf')
            return swaps
        
        # Possibility 1: do not swap the last element
        res1 = calculate(nums1[-1], nums2[-1])
        
        # Possibility 2: swap the last element
        res2 = calculate(nums2[-1], nums1[-1])
        
        min_ops = min(res1, res2)
        return min_ops if min_ops != float('inf') else -1