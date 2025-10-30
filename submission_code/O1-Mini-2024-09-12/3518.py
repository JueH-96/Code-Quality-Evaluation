from typing import List

class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        max0 = float('-inf')
        max1 = float('-inf')
        max2 = float('-inf')
        max3 = float('-inf')
        
        for num in b:
            # Update max3 based on the previous max2
            max3 = max(max3, max2 + a[3] * num)
            # Update max2 based on the previous max1
            max2 = max(max2, max1 + a[2] * num)
            # Update max1 based on the previous max0
            max1 = max(max1, max0 + a[1] * num)
            # Update max0
            max0 = max(max0, a[0] * num)
        
        return max3