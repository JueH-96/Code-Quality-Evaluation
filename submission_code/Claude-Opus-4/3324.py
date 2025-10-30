class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        
        # Count frequency of each element
        count = Counter(nums)
        
        # Check if any element appears more than twice
        for freq in count.values():
            if freq > 2:
                return False
        
        return True