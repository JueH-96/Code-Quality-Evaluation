from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # Get the sorted list of unique elements
        uniq = sorted(set(nums))
        uniq_len = len(uniq)
        
        # If there are no unique elements, return 0 (though constraints ensure len(nums) >= 1)
        if uniq_len == 0:
            return 0
        
        union_size = 0
        i = 0
        while i < uniq_len:
            # Start of the current interval
            current_start = uniq[i] - k
            current_end = uniq[i] + k
            
            # Extend the current interval to the right by merging overlapping intervals
            j = i + 1
            while j < uniq_len and (uniq[j] - k) <= current_end:
                current_end = max(current_end, uniq[j] + k)
                j += 1
            
            # Add the length of the merged interval to union size
            union_size += current_end - current_start + 1
            
            # Move to the start of the next non-overlapping interval
            i = j
        
        # The maximum number of distinct elements is the minimum of the total elements and the union size
        return min(len(nums), union_size)