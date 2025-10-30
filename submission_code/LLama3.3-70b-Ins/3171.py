from typing import List

class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate the sum of non-zero elements in both arrays
        sum1 = sum(num for num in nums1 if num != 0)
        sum2 = sum(num for num in nums2 if num != 0)
        
        # Calculate the difference between the sums of non-zero elements
        diff = abs(sum1 - sum2)
        
        # Count the number of zeros in both arrays
        zeros1 = nums1.count(0)
        zeros2 = nums2.count(0)
        
        # If the difference is greater than the total number of zeros, it's impossible to make the sums equal
        if diff > zeros1 + zeros2:
            return -1
        
        # If the difference is less than or equal to the total number of zeros, we can make the sums equal
        # We add the difference to the smaller sum and add 1 to the remaining zeros
        if sum1 < sum2:
            return sum1 + diff + (zeros1 - (diff % 2)) + (zeros2 - ((diff + 1) // 2))
        else:
            return sum2 + diff + (zeros2 - (diff % 2)) + (zeros1 - ((diff + 1) // 2))