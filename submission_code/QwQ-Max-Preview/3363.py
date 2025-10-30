from heapq import heappush, heappop
from collections import defaultdict
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        counts = dict()  # Maps each ID to its current count
        freq_count = defaultdict(int)  # Maps each count to the number of IDs with that count
        heap = []  # Max-heap implemented using min-heap with negated values
        ans = []
        
        for delta, num in zip(freq, nums):
            old_count = counts.get(num, 0)
            new_count = old_count + delta
            
            # Update the counts dictionary
            if new_count == 0:
                if num in counts:
                    del counts[num]
            else:
                counts[num] = new_count
            
            # Update freq_count for the old count
            if old_count > 0:
                freq_count[old_count] -= 1
                if freq_count[old_count] == 0:
                    del freq_count[old_count]
            
            # Update freq_count for the new count and push to heap if necessary
            if new_count > 0:
                freq_count[new_count] += 1
                heappush(heap, -new_count)
            
            # Find the current maximum frequency
            current_max = 0
            while heap:
                candidate = -heap[0]  # Check the current top of the heap
                if candidate in freq_count and freq_count[candidate] > 0:
                    current_max = candidate
                    break
                else:
                    heappop(heap)  # Remove invalid entries
            ans.append(current_max)
        
        return ans