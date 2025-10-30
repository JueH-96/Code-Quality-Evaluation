class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1_nonzero = 0
        count0s1 = 0
        for num in nums1:
            if num == 0:
                count0s1 += 1
            else:
                sum1_nonzero += num
                
        sum2_nonzero = 0
        count0s2 = 0
        for num in nums2:
            if num == 0:
                count0s2 += 1
            else:
                sum2_nonzero += num
                
        has_zero1 = count0s1 > 0
        has_zero2 = count0s2 > 0
        
        if not has_zero1 and not has_zero2:
            if sum1_nonzero == sum2_nonzero:
                return sum1_nonzero
            else:
                return -1
        elif has_zero1 and not has_zero2:
            sum_fixed = sum2_nonzero
            min_sum1 = sum1_nonzero + count0s1
            if min_sum1 > sum_fixed:
                return -1
            else:
                return sum_fixed
        elif not has_zero1 and has_zero2:
            sum_fixed = sum1_nonzero
            min_sum2 = sum2_nonzero + count0s2
            if min_sum2 > sum_fixed:
                return -1
            else:
                return sum_fixed
        else:
            min_sum1 = sum1_nonzero + count0s1
            min_sum2 = sum2_nonzero + count0s2
            return max(min_sum1, min_sum2)