class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        zeros1 = nums1.count(0)
        zeros2 = nums2.count(0)
        
        # If the total number of zeros in both arrays is greater than the absolute difference of their sums, it's possible to balance them
        if zeros1 + zeros2 < abs(sum1 - sum2):
            return -1
        
        # Calculate the minimum sum we can obtain
        total_sum = sum1 + sum2
        min_sum = (total_sum + abs(sum1 - sum2)) // 2
        
        return min_sum