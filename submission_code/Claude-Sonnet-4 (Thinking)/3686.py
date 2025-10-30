class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        
        def is_prefix(start1, len1, start2, len2):
            """Check if nums[start1:start1+len1] is a prefix of nums[start2:start2+len2]"""
            if len1 > len2:
                return False
            for i in range(len1):
                if nums[start1 + i] != nums[start2 + i]:
                    return False
            return True
        
        # Try all possible splits
        for i in range(1, n):  # nums1 ends at i-1, nums2 starts at i
            for j in range(i + 1, n):  # nums2 ends at j-1, nums3 starts at j
                # nums1 = nums[0:i], nums2 = nums[i:j], nums3 = nums[j:n]
                len1, len2, len3 = i, j - i, n - j
                
                # Check if nums1 is a prefix of nums2
                cond1 = is_prefix(0, len1, i, len2)
                
                # Check if nums2 is a prefix of nums3
                cond2 = is_prefix(i, len2, j, len3)
                
                if cond1 or cond2:
                    count += 1
        
        return count