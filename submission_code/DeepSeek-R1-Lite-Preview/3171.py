class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate sum and count of zeros for nums1
        sum1 = 0
        z1 = 0
        for num in nums1:
            if num == 0:
                z1 += 1
            else:
                sum1 += num
        
        # Calculate sum and count of zeros for nums2
        sum2 = 0
        z2 = 0
        for num in nums2:
            if num == 0:
                z2 += 1
            else:
                sum2 += num
        
        # Calculate minimal sums by replacing zeros with 1
        minimal_sum1 = sum1 + z1
        minimal_sum2 = sum2 + z2
        
        # Determine the minimal equal sum
        if minimal_sum1 == minimal_sum2:
            return minimal_sum1
        elif minimal_sum1 > minimal_sum2:
            if z2 > 0:
                return minimal_sum1
            else:
                return -1
        else:  # minimal_sum2 > minimal_sum1
            if z1 > 0:
                return minimal_sum2
            else:
                return -1