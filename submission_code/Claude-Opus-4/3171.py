class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate sum and count of zeros for both arrays
        sum1 = sum(nums1)
        zeros1 = nums1.count(0)
        
        sum2 = sum(nums2)
        zeros2 = nums2.count(0)
        
        # Minimum possible sums after replacing zeros with 1s
        min_sum1 = sum1 + zeros1
        min_sum2 = sum2 + zeros2
        
        # If both arrays have no zeros, they must already be equal
        if zeros1 == 0 and zeros2 == 0:
            return sum1 if sum1 == sum2 else -1
        
        # If one array has no zeros, it cannot change
        if zeros1 == 0:
            # nums1 is fixed at sum1
            # nums2 must be able to reach sum1
            if sum1 < min_sum2:
                return -1
            return sum1
        
        if zeros2 == 0:
            # nums2 is fixed at sum2
            # nums1 must be able to reach sum2
            if sum2 < min_sum1:
                return -1
            return sum2
        
        # Both arrays have zeros, so both can increase
        # The minimum equal sum is the maximum of their minimum possible sums
        return max(min_sum1, min_sum2)