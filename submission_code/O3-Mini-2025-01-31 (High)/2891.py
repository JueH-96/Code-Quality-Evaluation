from typing import List

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        # For each element, we can replace it with any value in [num - k, num + k].
        # To maximize the number of equal elements (beauty), we choose a target value X such that
        # as many elements as possible can be changed to X. This means that X must fall in as many
        # intervals [num-k, num+k] as possible.
        #
        # This is equivalent to finding a point X (an integer) which is contained in the maximum
        # number of these intervals. We can solve this with a sweep line algorithm.
        
        events = []
        for num in nums:
            # Each interval is [num - k, num + k] which means the element can be turned into any value
            # in that range.
            # We add an event at start (num - k) with +1 and at end+1 (num + k + 1) with -1.
            events.append((num - k, 1))
            events.append((num + k + 1, -1))
        
        events.sort()
        
        current_overlaps = 0
        max_overlaps = 0
        
        for point, delta in events:
            current_overlaps += delta
            max_overlaps = max(max_overlaps, current_overlaps)
            
        return max_overlaps