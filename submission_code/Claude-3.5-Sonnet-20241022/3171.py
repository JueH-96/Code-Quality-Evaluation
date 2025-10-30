class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        zeros1 = nums1.count(0)
        zeros2 = nums2.count(0)
        
        # If no zeros in one array but sums are different, impossible
        if zeros1 == 0 and sum1 < sum2 + zeros2:
            return -1
        if zeros2 == 0 and sum2 < sum1 + zeros1:
            return -1
            
        # Minimum possible sums after filling zeros
        min_sum1 = sum1 + zeros1  # Fill all zeros with 1
        min_sum2 = sum2 + zeros2  # Fill all zeros with 1
        
        # Return max of minimum possible sums
        return max(min_sum1, min_sum2)