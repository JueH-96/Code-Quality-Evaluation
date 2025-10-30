from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        freq = Counter(nums)
        if max(freq.values()) > 2:
            return False
        
        singles = 0
        for count in freq.values():
            if count == 1:
                singles += 1
                
        return singles % 2 == 0