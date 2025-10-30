class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        count0_1 = nums1.count(0)
        count0_2 = nums2.count(0)
        if count0_1 == 0 and count0_2 == 0:
            if sum1 == sum2:
                return sum1
            else:
                return -1
        elif count0_1 == 0:
            if sum1 >= sum2 + count0_2:
                return sum1
            else:
                return -1
        elif count0_2 == 0:
            if sum2 >= sum1 + count0_1:
                return sum2
            else:
                return -1
        else:
            return max(sum1 + count0_1, sum2 + count0_2)