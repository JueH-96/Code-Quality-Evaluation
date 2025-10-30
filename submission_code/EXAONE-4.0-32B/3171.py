class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        s1 = sum(nums1)
        s2 = sum(nums2)
        count1 = nums1.count(0)
        count2 = nums2.count(0)
        
        base1 = s1 + count1
        base2 = s2 + count2
        
        if count1 == 0 and count2 == 0:
            return s1 if s1 == s2 else -1
            
        if count1 == 0:
            if s1 >= base2:
                return s1
            else:
                return -1
                
        if count2 == 0:
            if s2 >= base1:
                return s2
            else:
                return -1
                
        return max(base1, base2)