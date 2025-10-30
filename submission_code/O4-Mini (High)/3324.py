from typing import List
from collections import Counter

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        # Count the occurrences of each number
        freq = Counter(nums)
        
        # If any number appears more than twice, one of the halves would
        # inevitably contain a duplicate of that number.
        if any(count > 2 for count in freq.values()):
            return False
        
        # We need each half to have nums.length/2 distinct elements.
        # Since no number appears more than twice, the maximum distinct
        # elements we can use is len(freq). We check if that's at least half.
        return len(freq) >= len(nums) // 2