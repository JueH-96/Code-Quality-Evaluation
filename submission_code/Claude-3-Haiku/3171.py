class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        sum1 = sum(nums1)
        sum2 = sum(nums2)
        
        if abs(sum1 - sum2) % 2 != 0:
            return -1
        
        target = (sum1 + sum2) // 2
        
        nums1_zeros = nums1.count(0)
        nums2_zeros = nums2.count(0)
        
        if nums1_zeros + nums2_zeros < abs(sum1 - sum2) // 2:
            return -1
        
        diff = abs(sum1 - sum2) // 2
        
        for i in range(nums1_zeros):
            if diff > 0:
                nums1[nums1.index(0)] = min(diff, 10**6 - min(nums1))
                diff -= nums1[nums1.index(0)]
        
        for i in range(nums2_zeros):
            if diff > 0:
                nums2[nums2.index(0)] = min(diff, 10**6 - min(nums2))
                diff -= nums2[nums2.index(0)]
        
        return target