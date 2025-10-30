import collections
import heapq
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        """
        Tracks the frequency of IDs in a collection that changes over time and returns the
        count of the most frequent ID at each step.
        """
        
        # id_counts maps an ID to its current frequency.
        # defaultdict(int) initializes a new key with a value of 0.
        id_counts = collections.defaultdict(int)
        
        # max_heap stores tuples of (-count, id) to find the max count quickly.
        # Python's heapq is a min-heap, so we use negative counts to simulate a max-heap.
        max_heap = []
        
        ans = []
        n = len(nums)

        for i in range(n):
            num = nums[i]
            fr = freq[i]

            # Update the frequency of the current ID.
            id_counts[num] += fr
            
            # Add the new frequency information to the heap.
            # This might add a duplicate ID with a new count, but we handle
            # these stale entries during the cleanup phase.
            heapq.heappush(max_heap, (-id_counts[num], num))

            # Lazily remove stale entries from the top of the heap.
            # A stale entry is one where the count in the heap does not match the current
            # actual count in id_counts. This happens when we update an ID's frequency.
            while max_heap and -max_heap[0][0] != id_counts[max_heap[0][1]]:
                heapq.heappop(max_heap)

            # After cleanup, the top of the heap is the most frequent ID.
            # If the heap is empty, it means all IDs have a count of 0,
            # so the collection is empty and the max frequency is 0.
            if not max_heap:
                ans.append(0)
            else:
                ans.append(-max_heap[0][0])
                
        return ans