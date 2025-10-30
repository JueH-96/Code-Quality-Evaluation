class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1_no_zeros = 0
        sum2_no_zeros = 0
        zeros1 = 0
        zeros2 = 0
        for x in nums1:
            if x == 0:
                zeros1 += 1
            else:
                sum1_no_zeros += x
        for x in nums2:
            if x == 0:
                zeros2 += 1
            else:
                sum2_no_zeros += x
        
        if sum1_no_zeros < sum2_no_zeros:
            if sum2_no_zeros + zeros2 < sum1_no_zeros + zeros1:
                return -1
            else:
                return sum2_no_zeros + zeros2
        elif sum1_no_zeros > sum2_no_zeros:
            if sum1_no_zeros + zeros1 < sum2_no_zeros + zeros2:
                return -1
            else:
                return sum1_no_zeros + zeros1
        else: # sum1_no_zeros == sum2_no_zeros
            return sum1_no_zeros + max(zeros1, zeros2)