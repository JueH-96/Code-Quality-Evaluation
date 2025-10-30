class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        for i in range(1, n):
            nums1 = nums[:i]
            for j in range(i+1, n):
                nums2 = nums[i:j]
                nums3 = nums[j:]
                
                # Check if nums1 is a prefix of nums2
                if self.is_prefix(nums1, nums2):
                    count += 1
                    continue
                
                # Check if nums2 is a prefix of nums3
                if self.is_prefix(nums2, nums3):
                    count += 1
        
        return count
    
    def is_prefix(self, prefix, array):
        if len(prefix) > len(array):
            return False
        for i in range(len(prefix)):
            if prefix[i] != array[i]:
                return False
        return True