class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        cnt1 = nums1.count(0)
        cnt2 = nums2.count(0)
        
        if cnt1 == 0 and cnt2 == 0:
            return sum1 if sum1 == sum2 else -1
        elif cnt1 == 0:
            if sum1 < sum2:
                return -1
            else:
                y = sum1 - sum2
                return sum1 if y >= cnt2 else -1
        elif cnt2 == 0:
            if sum2 < sum1:
                return -1
            else:
                x = sum2 - sum1
                return sum2 if x >= cnt1 else -1
        else:
            S1 = sum1 + cnt1
            S2 = sum2 + cnt2
            return max(S1, S2)