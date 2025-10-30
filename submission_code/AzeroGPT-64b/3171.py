from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        zero1, zero2, sum1, sum2 = 0, 0, 0, 0

        for num in nums1:
            if num == 0:
                zero1 += 1
            else:
                sum1 += num

        for num in nums2:
            if num == 0:
                zero2 += 1
            else:
                sum2 += num

        if sum1 + zero1 < sum2 and not zero2:
            return -1
        if zero1 and sum1 + zero1 == sum2:
            return sum1 + zero1
        if sum1 + zero1 > sum2:
            return sum1 + zero1

        if sum2 + zero2 < sum1 and not zero1:
            return -1
        if zero2 and sum2 + zero2 == sum1:
            return sum2 + zero2
        if sum2 + zero2 > sum1:
            return sum2 + zero2

        if sum1 < sum2 and not (not zero1 or not zero2 or (zero1 and not zero2) or (not zero1 and zero2)):
            return sum2
        if sum2 < sum1:
            return sum1

        return sum1 if sum1 == sum2 else -1