class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        s1 = sum(nums1)
        z1 = nums1.count(0)
        minSum1 = s1 + z1  # Since zeros are at least 1 when replaced

        s2 = sum(nums2)
        z2 = nums2.count(0)
        minSum2 = s2 + z2  # Same reasoning

        if minSum1 == minSum2:
            return minSum1
        elif minSum1 < minSum2:
            if z1 > 0:
                return minSum2
            else:
                return -1
        else:
            if z2 > 0:
                return minSum1
            else:
                return -1