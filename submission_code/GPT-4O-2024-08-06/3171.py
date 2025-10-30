class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        
        # Count the number of zeros in both arrays
        zeros1 = nums1.count(0)
        zeros2 = nums2.count(0)
        
        # Calculate the difference between the sums
        diff = abs(sum1 - sum2)
        
        # If the difference is greater than the number of zeros, it's impossible
        if diff > zeros1 + zeros2:
            return -1
        
        # Calculate the minimum sum we can achieve
        # We need to make the sums equal by replacing zeros with positive integers
        # The minimum sum will be the larger of the two sums plus the number of zeros
        # because we can replace each zero with 1 to achieve the minimum increase
        min_sum = max(sum1, sum2) + zeros1 + zeros2
        
        return min_sum