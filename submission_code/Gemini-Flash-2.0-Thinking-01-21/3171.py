class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = 0
        zeros1 = 0
        for num in nums1:
            if num == 0:
                zeros1 += 1
            else:
                sum1 += num
        sum2 = 0
        zeros2 = 0
        for num in nums2:
            if num == 0:
                zeros2 += 1
            else:
                sum2 += num
        
        s1 = sum1 + zeros1
        s2 = sum2 + zeros2
        
        if zeros1 == 0 and sum1 < s2:
            return -1
        if zeros2 == 0 and sum2 < s1:
            return -1
            
        return max(s1, s2)