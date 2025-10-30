class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate current sums and count zeros
        sum1 = sum(nums1)
        zeros1 = nums1.count(0)
        sum2 = sum(nums2)
        zeros2 = nums2.count(0)
        
        # Minimum possible sum for each array (replacing 0s with 1s)
        min_sum1 = sum1 + zeros1
        min_sum2 = sum2 + zeros2
        
        # Case 1: Both arrays have no zeros - sums are fixed
        if zeros1 == 0 and zeros2 == 0:
            return sum1 if sum1 == sum2 else -1
        
        # Case 2: Only nums1 has no zeros - its sum is fixed
        elif zeros1 == 0:
            # nums2 needs to reach sum1, check if possible
            return sum1 if sum1 >= min_sum2 else -1
        
        # Case 3: Only nums2 has no zeros - its sum is fixed
        elif zeros2 == 0:
            # nums1 needs to reach sum2, check if possible
            return sum2 if sum2 >= min_sum1 else -1
        
        # Case 4: Both arrays have zeros - both are flexible
        else:
            # Return the maximum of the minimum possible sums
            return max(min_sum1, min_sum2)