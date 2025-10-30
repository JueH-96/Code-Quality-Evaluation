from typing import List
from collections import defaultdict
import heapq

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        # id_count maps each ID to its current count in the collection
        id_count = {}
        # count_of_counts maps a count -> how many IDs currently have that count
        count_of_counts = defaultdict(int)
        # max_heap is a max-heap (implemented via heapq on negated counts)
        max_heap = []
        ans = []
        
        for x, delta in zip(nums, freq):
            old_count = id_count.get(x, 0)
            new_count = old_count + delta
            
            # Remove the old count from count_of_counts
            if old_count > 0:
                count_of_counts[old_count] -= 1
            
            # If after the update the ID still has positive occurrences
            if new_count > 0:
                id_count[x] = new_count
                count_of_counts[new_count] += 1
                # Push the new count onto the heap (negated for max-heap)
                heapq.heappush(max_heap, -new_count)
            else:
                # new_count == 0 => ID is gone from the collection
                if x in id_count:
                    del id_count[x]
            
            # Clean up the heap top if it refers to a count that's no longer present
            while max_heap and count_of_counts[-max_heap[0]] == 0:
                heapq.heappop(max_heap)
            
            # The current maximum frequency is at heap top (or 0 if empty)
            ans.append(-max_heap[0] if max_heap else 0)
        
        return ans