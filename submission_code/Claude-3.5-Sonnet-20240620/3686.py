class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        beautiful_splits = 0
        
        def is_prefix(arr1, arr2):
            return arr1 == arr2[:len(arr1)]
        
        for i in range(1, n - 1):  # First split point
            for j in range(i + 1, n):  # Second split point
                nums1 = nums[:i]
                nums2 = nums[i:j]
                nums3 = nums[j:]
                
                if is_prefix(nums1, nums2) or is_prefix(nums2, nums3):
                    beautiful_splits += 1
        
        return beautiful_splits