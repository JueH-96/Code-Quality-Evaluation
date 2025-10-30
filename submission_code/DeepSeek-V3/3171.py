class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        count1 = nums1.count(0)
        count2 = nums2.count(0)
        
        # If both arrays have no zeros, their sums must be equal
        if count1 == 0 and count2 == 0:
            return sum1 if sum1 == sum2 else -1
        
        # If one array has no zeros and the other has, it's impossible unless the sum of the non-zero array is greater than or equal to the sum of the other array plus the number of zeros
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
        
        # Calculate the minimum possible sum
        # The sum after replacing zeros must be equal
        # sum1 + x * count1 = sum2 + y * count2
        # We need to find x and y such that x >= 1 and y >= 1
        # To minimize the sum, set x and y to 1
        # So sum1 + count1 = sum2 + count2
        # If sum1 + count1 == sum2 + count2, then the sum is sum1 + count1
        # Otherwise, it's impossible
        if (sum1 + count1) == (sum2 + count2):
            return sum1 + count1
        else:
            return -1