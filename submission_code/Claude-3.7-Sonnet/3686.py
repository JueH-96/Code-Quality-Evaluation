class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Try all possible combinations of splits
        for i in range(1, n-1):  # First split after i elements
            for j in range(i+1, n):  # Second split after j elements
                nums1 = nums[:i]
                nums2 = nums[i:j]
                nums3 = nums[j:]
                
                # Check if nums1 is a prefix of nums2
                is_prefix_1_2 = (len(nums1) <= len(nums2)) and (nums1 == nums2[:len(nums1)])
                
                # Check if nums2 is a prefix of nums3
                is_prefix_2_3 = (len(nums2) <= len(nums3)) and (nums2 == nums3[:len(nums2)])
                
                if is_prefix_1_2 or is_prefix_2_3:
                    count += 1
        
        return count