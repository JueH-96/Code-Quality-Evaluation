from typing import List
import collections

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        freq = collections.Counter(nums)
        max_freq = max(freq.values())
        
        if max_freq <= n // 2:
            return n % 2
        else:
            return 2 * max_freq - n