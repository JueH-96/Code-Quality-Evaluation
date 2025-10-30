import heapq
from collections import defaultdict
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        counts = defaultdict(int)
        freq_counter = defaultdict(int)
        heap = []
        ans = []
        
        for i in range(len(nums)):
            ID = nums[i]
            delta = freq[i]
            current_count = counts[ID]
            new_count = current_count + delta
            
            # Update the old frequency in freq_counter
            if current_count > 0:
                freq_counter[current_count] -= 1
                if freq_counter[current_count] == 0:
                    del freq_counter[current_count]
            
            # Update the count for the current ID
            counts[ID] = new_count
            
            # Update the new frequency in freq_counter if it's positive
            if new_count > 0:
                freq_counter[new_count] += 1
                heapq.heappush(heap, -new_count)
            
            # Find the current maximum frequency
            current_max = 0
            while heap:
                candidate = -heap[0]
                if freq_counter.get(candidate, 0) > 0:
                    current_max = candidate
                    break
                else:
                    heapq.heappop(heap)
            
            ans.append(current_max)
        
        return ans