class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Iterate over possible split points for nums1 and nums2
        for i in range(1, n):
            # nums1 = nums[:i]
            # Iterate over possible split points for nums2 and nums3
            for j in range(i, n):
                # nums2 = nums[i:j]
                # nums3 = nums[j:]
                
                # Check if nums1 is a prefix of nums2
                if nums[:i] == nums[i:j][:i]:
                    count += 1
                
                # Check if nums2 is a prefix of nums3
                if nums[i:j] == nums[j:j+(j-i)]:
                    count += 1
        
        return count