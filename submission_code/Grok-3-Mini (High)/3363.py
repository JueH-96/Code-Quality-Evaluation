import heapq
import collections
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        freq_count = collections.defaultdict(int)
        heap = []  # min-heap storing (-frequency, id)
        ans = []
        
        for i in range(len(nums)):
            id_val = nums[i]
            change = freq[i]
            # Update the frequency count for the ID
            freq_count[id_val] += change
            new_freq = freq_count[id_val]
            
            # Add the new frequency to the heap only if positive
            if new_freq > 0:
                heapq.heappush(heap, (-new_freq, id_val))
            
            # Clean up invalid entries in the heap
            while heap and -heap[0][0] != freq_count[heap[0][1]]:
                heapq.heappop(heap)
            
            # Determine the maximum frequency after cleanup
            if heap:
                max_freq = -heap[0][0]
            else:
                max_freq = 0
            
            # Record the result for this step
            ans.append(max_freq)
        
        return ans