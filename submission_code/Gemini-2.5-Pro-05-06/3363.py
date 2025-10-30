import heapq
from collections import Counter
from typing import List, Tuple # Ensure List and Tuple are imported for type hints

class Solution:
  def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
    n = len(nums)
    ans = [0] * n  # Initialize answers with 0, for steps where collection is empty
    
    id_counts = Counter()
    # min_heap stores tuples of (-count, id) to simulate a max-heap of counts.
    # We store -count because heapq is a min-heap.
    # The ID is stored to check for staleness.
    min_heap: List[Tuple[int, int]] = [] # type hint for min_heap
    
    for i in range(n):
      num = nums[i]
      f = freq[i]
      
      # Update the count of the current ID in our frequency map.
      # Counter behavior:
      # - If num is not in id_counts, id_counts[num] is effectively 0 before this operation.
      #   So, id_counts[num] becomes 0 + f = f.
      # - If num is in id_counts, its current count is updated by f.
      # - If id_counts[num] becomes 0 (e.g., was 2, f is -2), Counter stores it as {num: 0}.
      #   The key `num` remains in `id_counts`. This is important for correctness of
      #   `id_counts[id_in_heap]` access later.
      # The problem guarantees counts will not become negative.
      id_counts[num] += f
      
      # Add the new (or updated) count to the heap.
      # The element pushed is (-actual_new_count, num_id).
      # actual_new_count is id_counts[num] after the update.
      heapq.heappush(min_heap, (-id_counts[num], num))
      
      # Clean the heap to find the current most frequent ID's count.
      # We need to remove/ignore:
      # 1. Stale entries: entries in heap whose count for an ID does not match
      #    the ID's current actual count in id_counts. This happens because we
      #    don't remove old entries from heap, only add new ones.
      # 2. Entries for IDs whose actual count is 0. These IDs are not considered
      #    part of the "collection" for finding most frequent.
      while min_heap:
        # Peek at the top element (smallest -count, so largest count)
        neg_count_in_heap, id_in_heap = min_heap[0]
        
        # Retrieve the actual current count of this ID from our map.
        # This access is safe because `id_in_heap` must be in `id_counts` (see notes in thought process).
        actual_current_count_of_id = id_counts[id_in_heap]
        
        # Check for staleness.
        # If actual_current_count_of_id is different from what's stored in heap (-neg_count_in_heap),
        # then this heap entry is stale (it represents a past count of id_in_heap).
        if actual_current_count_of_id != -neg_count_in_heap:
          heapq.heappop(min_heap)  # Pop stale entry and check next.
          continue
        
        # At this point, the heap entry is not stale: actual_current_count_of_id == -neg_count_in_heap.
        # Now, check if this valid count is 0.
        # By problem constraints, counts are non-negative, so actual_current_count_of_id >= 0.
        if actual_current_count_of_id == 0:
          # This ID's count is 0. It's not "in the collection" effectively.
          # We are interested in positive frequencies. So, pop this (0, id) entry.
          heapq.heappop(min_heap) # Pop this entry and check next.
          continue
          
        # If we reach here, the top heap element is valid (not stale),
        # and its count is positive. This is the current maximum frequency.
        ans[i] = actual_current_count_of_id
        break # Found the max for current step i.
      
      # If the while loop finishes (min_heap became empty) without `break`,
      # it means all IDs have count 0 or the collection is effectively empty.
      # In this case, ans[i] remains 0 (as initialized when ans array was created).
      
    return ans