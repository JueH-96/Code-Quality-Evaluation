from bisect import bisect_left, bisect_right
from collections import Counter
from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        # If no operations allowed or no "move" range, answer is just maximum frequency already.
        if k == 0 or numOperations == 0:
            return Counter(nums).most_common(1)[0][1]
        
        nums.sort()
        freq = Counter(nums)
        unique = sorted(freq.keys())
        n = len(nums)
        best = 0
        
        # For each candidate t (which is already in the array),
        # determine how many other elements can be converted into t
        # (i.e. each other x that lies in the range [t - k, t + k])
        for t in unique:
            # Using binary search on the sorted overall nums,
            # find the indices of numbers in [t-k, t+k]
            left = bisect_left(nums, t - k)
            right = bisect_right(nums, t + k)
            
            total_in_range = right - left  # count of nums that are convertible to t
            same = freq[t]
            convertible = total_in_range - same  # others that can be converted with one op each
            
            # We can only convert at most numOperations elements.
            candidate_freq = same + min(numOperations, convertible)
            best = max(best, candidate_freq)
            
        return best