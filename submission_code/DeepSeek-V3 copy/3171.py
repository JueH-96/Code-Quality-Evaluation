class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        count1 = nums1.count(0)
        count2 = nums2.count(0)
        
        # If both arrays have no zeros, their sums must be equal
        if count1 == 0 and count2 == 0:
            return sum1 if sum1 == sum2 else -1
        
        # If one array has no zeros and the other has zeros, it's impossible unless the sum of the non-zero array is greater than or equal to the sum of the other array plus the number of zeros
        if count1 == 0:
            if sum1 < sum2 + count2:
                return -1
            else:
                return sum1
        if count2 == 0:
            if sum2 < sum1 + count1:
                return -1
            else:
                return sum2
        
        # Both arrays have zeros
        # The minimum sum is the maximum of (sum1 + count1) and (sum2 + count2)
        min_sum = max(sum1 + count1, sum2 + count2)
        return min_sum