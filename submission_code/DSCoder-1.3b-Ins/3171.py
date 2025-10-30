class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        nums1.sort()
        nums2.sort()
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        if sum1 < sum2:
            return sum1
        else:
            return sum2