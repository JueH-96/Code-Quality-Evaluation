class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        
        freq = Counter(nums)
        
        # If any element appears more than twice, it's impossible
        # to split into two "distinct-element" parts of equal size.
        for count in freq.values():
            if count > 2:
                return False
        
        # Otherwise, it is always possible to distribute the elements:
        # - Each element that appears twice can go one in each part.
        # - All single-appearance elements can be chosen freely to fulfill the size requirements.
        return True