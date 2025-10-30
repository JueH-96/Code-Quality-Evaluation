class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate sum of non-zero elements and count zeros
        sum1 = sum(x for x in nums1 if x != 0)
        sum2 = sum(x for x in nums2 if x != 0)
        zeros1 = nums1.count(0)
        zeros2 = nums2.count(0)
        
        # Minimum possible sum for each array
        min_sum1 = sum1 + zeros1  # Each 0 replaced with at least 1
        min_sum2 = sum2 + zeros2  # Each 0 replaced with at least 1
        
        # If neither array has zeros, sums are fixed
        if zeros1 == 0 and zeros2 == 0:
            return sum1 if sum1 == sum2 else -1
        
        # If only one array has zeros
        if zeros1 == 0:  # nums1 sum is fixed
            return sum1 if sum1 >= min_sum2 else -1
        
        if zeros2 == 0:  # nums2 sum is fixed
            return sum2 if sum2 >= min_sum1 else -1
        
        # Both arrays have zeros - return the maximum of their minimum sums
        return max(min_sum1, min_sum2)