class Solution:
    def minSum(self, nums1: list[int], nums2: list[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        zeros1 = nums1.count(0)
        zeros2 = nums2.count(0)
        
        # If both arrays have no zeros, check if their sums are already equal
        if zeros1 == 0 and zeros2 == 0:
            return sum1 if sum1 == sum2 else -1
        
        # If one array has zeros and the other has a larger sum without zeros, it's impossible
        if (zeros1 > 0 and sum1 > sum2) or (zeros2 > 0 and sum2 > sum1):
            return -1
        
        # Calculate the difference in sums and the total number of zeros
        diff = abs(sum1 - sum2)
        total_zeros = zeros1 + zeros2
        
        # If the difference is not divisible by the total number of zeros, it's impossible
        if diff % total_zeros != 0:
            return -1
        
        # Calculate the value to fill each zero with
        fill_value = diff // total_zeros
        
        # If the fill value is zero, it means the sums were already equal, so we can fill with 1
        if fill_value == 0:
            fill_value = 1
        
        # Calculate the new sum after filling zeros
        new_sum = sum1 + zeros1 * fill_value
        
        return new_sum