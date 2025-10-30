class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        
        # Count frequency of each element
        count = Counter(nums)
        
        # If any element appears more than 2 times, 
        # we cannot split into two distinct arrays
        for freq in count.values():
            if freq > 2:
                return False
        
        return True