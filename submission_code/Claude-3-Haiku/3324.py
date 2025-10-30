class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        n = len(nums)
        if n % 2 != 0:
            return False
        
        nums.sort()
        nums1 = []
        nums2 = []
        
        for i in range(0, n, 2):
            if nums[i] not in nums1:
                nums1.append(nums[i])
            else:
                if nums[i+1] not in nums2:
                    nums2.append(nums[i+1])
                else:
                    return False
        
        return True