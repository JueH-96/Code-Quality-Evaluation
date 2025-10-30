class Solution:
    def minSum(self, nums1, nums2):
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        zeros1 = nums1.count(0)
        zeros2 = nums2.count(0)
        diff = abs(sum1 - sum2)
        if diff % (zeros1 + zeros2) != 0:
            return -1
        else:
            return max(sum1, sum2) + diff // 2