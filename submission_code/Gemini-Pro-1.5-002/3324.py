class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        n = len(nums)
        
        for i in range(1 << n):
            nums1 = []
            nums2 = []
            for j in range(n):
                if (i >> j) & 1:
                    nums1.append(nums[j])
                else:
                    nums2.append(nums[j])
            
            if len(nums1) == n // 2:
                if len(set(nums1)) == n // 2 and len(set(nums2)) == n // 2:
                    return True
        
        return False