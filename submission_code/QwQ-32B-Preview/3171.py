class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        # Calculate sum and count of zeros for nums1
        sum1 = sum(num for num in nums1 if num > 0)
        count1 = nums1.count(0)
        
        # Calculate sum and count of zeros for nums2
        sum2 = sum(num for num in nums2 if num > 0)
        count2 = nums2.count(0)
        
        # Calculate the minimum equal sum S
        S = max(sum1 + count1, sum2 + count2)
        
        # Check feasibility
        if count1 == 0 and sum1 != S:
            return -1
        if count2 == 0 and sum2 != S:
            return -1
        
        return S