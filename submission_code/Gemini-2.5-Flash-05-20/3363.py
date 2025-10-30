import collections
import heapq
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        n = len(nums)
        ans = [0] * n
        
        # id_counts: A dictionary to store the current total count for each ID.
        # Keys are IDs (int), values are their current total frequencies (int).
        # collections.defaultdict(int) conveniently initializes new ID counts to 0.
        id_counts = collections.defaultdict(int)
        
        # max_heap: A min-heap to efficiently track the maximum frequency.
        # Elements are stored as (-count, id) tuples.
        # By storing negative counts, the smallest element in the min-heap
        # corresponds to the largest actual count.
        # This heap may contain "stale" entries, which are lazily removed.
        # A stale entry is one where the 'count' in the tuple (-count, id)
        # does not match the actual current count of 'id' in id_counts.
        max_heap = [] 
        
        for i in range(n):
            current_id = nums[i]
            change_in_freq = freq[i]
            
            # 1. Update the current total count for 'current_id'.
            # The problem guarantees that id_counts[current_id] will not become negative.
            id_counts[current_id] += change_in_freq
            new_count = id_counts[current_id]
            
            # 2. Add the potentially new maximum frequency to the heap.
            # We push (-new_count, current_id) to maintain a min-heap structure
            # while effectively tracking maximum counts.
            heapq.heappush(max_heap, (-new_count, current_id))
            
            # 3. Determine the actual current most frequent ID's count.
            # We peek at the top of the heap, which represents the largest count (smallest negative).
            # If this entry is stale (its stored count doesn't match id_counts),
            # we pop it and repeat until a valid entry is found or the heap is empty.
            while max_heap:
                # Peek at the element with the highest reported frequency (smallest negative count)
                top_neg_count, top_id = max_heap[0] 
                
                # Get the actual current count for 'top_id' from id_counts.
                actual_count = id_counts[top_id]
                
                # Check if the heap entry is still valid (not stale).
                if actual_count == -top_neg_count:
                    # This entry is valid: its count matches the current actual count.
                    # This is the highest frequency currently present in the collection.
                    ans[i] = actual_count
                    break # Found the max, exit the while loop
                else:
                    # This entry is stale (its count in the heap is outdated).
                    # Lazily remove it by popping from the heap.
                    heapq.heappop(max_heap)
            
            # 4. Handle the case where the collection becomes empty.
            # If the while loop above exhausts the heap (meaning all entries
            # were stale or the heap was empty initially), it implies there
            # are no IDs with positive counts in the collection.
            if not max_heap:
                ans[i] = 0 # Collection is empty, so most frequent count is 0.
                
        return ans