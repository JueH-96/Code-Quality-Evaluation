class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        # Try all possible split points
        # i is the end of nums1 (exclusive), j is the end of nums2 (exclusive)
        for i in range(1, n):  # nums1 must have at least 1 element
            for j in range(i + 1, n):  # nums2 must have at least 1 element, nums3 must have at least 1 element
                # nums1 = nums[0:i]
                # nums2 = nums[i:j]
                # nums3 = nums[j:]
                
                # Check if nums1 is a prefix of nums2
                len1 = i
                len2 = j - i
                
                is_prefix_1_2 = False
                if len1 <= len2:
                    is_prefix_1_2 = True
                    for k in range(len1):
                        if nums[k] != nums[i + k]:
                            is_prefix_1_2 = False
                            break
                
                # Check if nums2 is a prefix of nums3
                len3 = n - j
                
                is_prefix_2_3 = False
                if len2 <= len3:
                    is_prefix_2_3 = True
                    for k in range(len2):
                        if nums[i + k] != nums[j + k]:
                            is_prefix_2_3 = False
                            break
                
                # If either condition is satisfied, it's a beautiful split
                if is_prefix_1_2 or is_prefix_2_3:
                    count += 1
        
        return count