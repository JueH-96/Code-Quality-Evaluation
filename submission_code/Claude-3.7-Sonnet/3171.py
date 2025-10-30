class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(num for num in nums1 if num != 0)
        sum2 = sum(num for num in nums2 if num != 0)
        zeros1 = nums1.count(0)
        zeros2 = nums2.count(0)
        
        min_sum1 = sum1 + zeros1  # Minimum possible sum for nums1
        min_sum2 = sum2 + zeros2  # Minimum possible sum for nums2
        
        # If both arrays have no zeros
        if zeros1 == 0 and zeros2 == 0:
            return sum1 if sum1 == sum2 else -1
        
        # If nums1 has no zeros but minimum sum of nums2 is greater
        if zeros1 == 0 and min_sum1 < min_sum2:
            return -1
            
        # If nums2 has no zeros but minimum sum of nums1 is greater
        if zeros2 == 0 and min_sum2 < min_sum1:
            return -1
        
        # Return the maximum of the two minimum sums
        return max(min_sum1, min_sum2)