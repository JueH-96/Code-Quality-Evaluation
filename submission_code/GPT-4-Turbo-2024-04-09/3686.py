class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # We need to check for every possible split point i and j where 0 <= i < j < n
        for i in range(1, n-1):
            for j in range(i+1, n):
                nums1 = nums[:i]
                nums2 = nums[i:j]
                nums3 = nums[j:]
                
                # Check if nums1 is a prefix of nums2
                if len(nums1) <= len(nums2) and nums1 == nums2[:len(nums1)]:
                    count += 1
                # Check if nums2 is a prefix of nums3
                elif len(nums2) <= len(nums3) and nums2 == nums3[:len(nums2)]:
                    count += 1
        
        return count