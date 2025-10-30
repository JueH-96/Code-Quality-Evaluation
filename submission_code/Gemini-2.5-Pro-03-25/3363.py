import heapq
from collections import defaultdict
from typing import List

class Solution:
  """
  Solves the problem of finding the most frequent ID in a collection that changes over time.
  Uses a frequency map and a max-heap to efficiently track the maximum frequency at each step.
  """
  def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
    """
    Calculates the count of the most frequent ID after each step defined by the nums and freq arrays.

    Args:
        nums: A list of integers representing IDs.
        freq: A list of integers representing frequency changes for the corresponding IDs in nums.
              A positive value indicates additions (freq[i] IDs with value nums[i] are added).
              A negative value indicates removals (-freq[i] IDs with value nums[i] are removed).

    Returns:
        A list of integers `ans` where `ans[i]` is the count of the most frequent ID in the collection 
        after the i-th step. If the collection is empty at step i, `ans[i]` is 0.
    
    The approach uses:
    1. A dictionary `id_freq` (using `defaultdict(int)`) to maintain the current frequency count of each ID.
    2. A max-heap `freq_heap` to efficiently retrieve the maximum frequency currently present in the collection. 
       Since Python's `heapq` implements a min-heap, we store tuples `(-frequency, ID)` to simulate a max-heap 
       ordered by frequency.

    Algorithm:
    Iterate through each step `i` from 0 to n-1:
    a. Get the ID `id_val = nums[i]` and its frequency change `f_change = freq[i]`.
    b. Update the frequency of `id_val` in the `id_freq` dictionary: `id_freq[id_val] += f_change`. Let the new frequency be `new_freq`.
    c. If `new_freq` becomes 0, it means the ID is no longer present in the collection with a positive count. Remove the entry for `id_val` from `id_freq`. The problem constraints guarantee the frequency will not become negative.
    d. If `new_freq` is positive, push the current state `(-new_freq, id_val)` onto the `freq_heap`. This adds the ID's current frequency to the heap for max frequency tracking. Note that older entries for the same ID might still exist in the heap but are considered 'stale'.
    e. Clean the `freq_heap`: While the heap is not empty, check the top element `(-heap_freq, heap_id)`. If the frequency `heap_freq` does not match the actual current frequency of `heap_id` stored in `id_freq` (or if `heap_id` is no longer in `id_freq`), this entry is stale. Pop it from the heap and repeat. Continue until the top element is valid or the heap is empty.
    f. After cleaning, if `freq_heap` is not empty, the maximum frequency is given by the frequency component of the top element (`-freq_heap[0][0]`). Store this in `ans[i]`.
    g. If `freq_heap` is empty after cleaning, it means the collection is currently empty (all IDs have zero frequency). Store 0 in `ans[i]`.
    Finally, return the `ans` array.

    Time Complexity: O(n log n), where n is the number of steps (length of nums/freq).
                     Each step involves dictionary operations (O(1) average time), a heap push (O(log k), where k is heap size, k <= n),
                     and potentially multiple heap pops during the cleanup phase. However, each element is pushed onto the heap at most once and popped at most once over the entire process.
                     Therefore, the total time complexity is dominated by the heap operations, resulting in O(n log n).
    Space Complexity: O(n) in the worst case.
                      `id_freq` can store up to n distinct IDs.
                      `freq_heap` can store up to n entries if every step adds a new frequency state.
                      The result array `ans` also requires O(n) space.
    """
    
    n = len(nums)
    # Initialize the result array to store the max frequency at each step.
    ans = [0] * n  
    
    # id_freq: A dictionary to store the current frequency of each ID.
    # defaultdict(int) automatically initializes frequency to 0 for new IDs.
    id_freq = defaultdict(int) 
    
    # freq_heap: A max-heap (simulated using min-heap) storing tuples (-frequency, ID).
    # This allows O(log n) retrieval of the maximum frequency after cleanup.
    freq_heap = [] 
    
    for i in range(n):
        id_val = nums[i]    # The ID processed in this step.
        f_change = freq[i]  # The frequency change for this ID.
        
        # Update the frequency of id_val in the dictionary.
        id_freq[id_val] += f_change
        new_freq = id_freq[id_val]
        
        # If the frequency becomes zero, remove the ID's entry from the frequency map.
        # According to constraints, frequencies won't become negative.
        if new_freq == 0:
            # Ensure the key is removed if its count reaches zero.
            if id_val in id_freq: # Necessary check before deleting from defaultdict context
               del id_freq[id_val]
        
        # If the new frequency is positive, push its current state onto the max-heap.
        # This tuple represents the current frequency of the ID. Stale entries are handled later.
        if new_freq > 0:
             heapq.heappush(freq_heap, (-new_freq, id_val))
        
        # Clean the top of the heap from stale entries.
        # A stale entry has a frequency that doesn't match the ID's current frequency in id_freq.
        while freq_heap:
            # Peek at the element with the highest frequency (smallest negative frequency).
            neg_max_freq, max_id = freq_heap[0] 
            
            # Get the current, actual frequency of this ID from the map.
            # Use .get(max_id, 0) which returns 0 if max_id is not in id_freq (i.e., freq is 0).
            current_actual_freq = id_freq.get(max_id, 0) 
            
            # Check if the heap entry's frequency (-neg_max_freq) matches the actual frequency.
            if current_actual_freq == -neg_max_freq:
                # If they match, the top element is valid and represents the current max frequency.
                break # Stop cleaning
            else:
                # If they don't match, the heap entry is stale. Pop it and check the next one.
                heapq.heappop(freq_heap)
        
        # After cleaning, if the heap is not empty, the top element gives the maximum frequency.
        if freq_heap:
            # The maximum frequency is -neg_max_freq.
            ans[i] = -freq_heap[0][0] 
        else:
            # If the heap is empty, it means no IDs have positive frequency in the collection.
            ans[i] = 0
            
    return ans