class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        count1 = nums1.count(0)
        count2 = nums2.count(0)

        if count1 == 0 and count2 == 0:
            if sum1 == sum2:
                return sum1
            else:
                return -1
        elif count1 == 0:
            if sum1 < sum2 + count2:
                return -1
            else:
                if sum1 == sum2 + count2:
                    return sum1
                else:
                    return max(sum1, sum2 + count2)
        elif count2 == 0:
            if sum2 < sum1 + count1:
                return -1
            else:
                if sum2 == sum1 + count1:
                    return sum2
                else:
                    return max(sum1 + count1, sum2)
        else:
            if sum1 + count1 < sum2 + count2 and sum1 + count1 != sum2 + count2:
                return -1
            elif sum2 + count2 < sum1 + count1 and sum1 + count1 != sum2 + count2:
                return -1
            else:
                return max(sum1 + count1, sum2 + count2)