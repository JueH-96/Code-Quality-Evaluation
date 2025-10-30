class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        from collections import Counter
        
        count = Counter(nums)
        unique_count = len(count)
        
        # If k is greater than or equal to the number of unique elements minus 1,
        # we can take all elements.
        if k >= unique_count - 1:
            return len(nums)
        
        # Otherwise, we can take all but (k + 1) unique elements.
        return len(nums) - (unique_count - (k + 1))