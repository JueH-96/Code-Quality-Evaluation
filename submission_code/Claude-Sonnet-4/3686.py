class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Try all possible split points
        for i in range(n - 2):  # i is the last index of nums1
            for j in range(i + 1, n - 1):  # j is the last index of nums2
                nums1 = nums[0:i+1]
                nums2 = nums[i+1:j+1]
                nums3 = nums[j+1:]
                
                # Check if nums1 is prefix of nums2
                is_nums1_prefix_of_nums2 = False
                if len(nums1) <= len(nums2):
                    is_nums1_prefix_of_nums2 = nums2[:len(nums1)] == nums1
                
                # Check if nums2 is prefix of nums3
                is_nums2_prefix_of_nums3 = False
                if len(nums2) <= len(nums3):
                    is_nums2_prefix_of_nums3 = nums3[:len(nums2)] == nums2
                
                # If either condition is satisfied, it's a beautiful split
                if is_nums1_prefix_of_nums2 or is_nums2_prefix_of_nums3:
                    count += 1
        
        return count