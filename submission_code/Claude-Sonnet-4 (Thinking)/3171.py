class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        zeros1 = nums1.count(0)
        zeros2 = nums2.count(0)
        
        min_sum1 = sum1 + zeros1
        min_sum2 = sum2 + zeros2
        
        if zeros1 == 0 and zeros2 == 0:
            return sum1 if sum1 == sum2 else -1
        elif zeros1 == 0:
            return sum1 if min_sum2 <= sum1 else -1
        elif zeros2 == 0:
            return sum2 if min_sum1 <= sum2 else -1
        else:
            return max(min_sum1, min_sum2)