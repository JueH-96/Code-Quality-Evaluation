class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        
        count = Counter(nums)
        distinct_count = len(count)
        total_elements = len(nums)
        
        # We need at least half of the distinct elements to fill nums1 and nums2
        return distinct_count >= total_elements // 2