from typing import List
from collections import Counter

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        # Count the frequency of each element in the array
        frequency = Counter(nums)

        # Count the number of unique elements that appear exactly once
        unique_count = sum(1 for count in frequency.values() if count == 1)

        return unique_count