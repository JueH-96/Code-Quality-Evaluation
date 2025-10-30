class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        max0 = float('-inf')
        max1 = float('-inf')
        max2 = float('-inf')
        max3 = float('-inf')
        for num in b:
            # Update max3 using the previous max2
            candidate3 = max2 + a[3] * num
            if candidate3 > max3:
                max3 = candidate3
            # Update max2 using the previous max1
            candidate2 = max1 + a[2] * num
            if candidate2 > max2:
                max2 = candidate2
            # Update max1 using the previous max0
            candidate1 = max0 + a[1] * num
            if candidate1 > max1:
                max1 = candidate1
            # Update max0 using the current number
            candidate0 = a[0] * num
            if candidate0 > max0:
                max0 = candidate0
        return max3