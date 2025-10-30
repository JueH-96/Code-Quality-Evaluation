class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = 0
        c1 = 0
        for num in nums1:
            if num != 0:
                sum1 += num
            else:
                c1 += 1
        
        sum2 = 0
        c2 = 0
        for num in nums2:
            if num != 0:
                sum2 += num
            else:
                c2 += 1
        
        if c1 == 0 and c2 == 0:
            return sum1 if sum1 == sum2 else -1
        elif c1 == 0:
            required_sum = sum1
            min_sum2 = sum2 + c2
            return required_sum if required_sum >= min_sum2 else -1
        elif c2 == 0:
            required_sum = sum2
            min_sum1 = sum1 + c1
            return required_sum if required_sum >= min_sum1 else -1
        else:
            sum1_min = sum1 + c1
            sum2_min = sum2 + c2
            return max(sum1_min, sum2_min)