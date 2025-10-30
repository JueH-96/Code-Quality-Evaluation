class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter

        freq = Counter(nums)
        # If any element appears more than twice, it's impossible
        # to split into two parts where each part has distinct elements.
        for count in freq.values():
            if count > 2:
                return False
        
        return True