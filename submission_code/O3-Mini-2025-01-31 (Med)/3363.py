from collections import Counter
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        # Dictionary to keep track of the current count of each ID.
        id_count = {}
        # Counter for how many IDs have a given count.
        # We maintain counts only for positive counts (IDs in the collection).
        freq_count = Counter()
        # This variable tracks the current maximum frequency in the collection.
        max_freq = 0
        ans = []
        
        for num, change in zip(nums, freq):
            # Get the old count if it exists, otherwise 0.
            old = id_count.get(num, 0)
            new = old + change  # new count after the update (guaranteed non-negative)
            
            # If the id had a positive count, update the frequency counter for its old count.
            if old > 0:
                freq_count[old] -= 1
                if freq_count[old] == 0:
                    del freq_count[old]
            
            # Update id_count: 
            # If new count is 0, remove it (the id is not in the collection anymore).
            # Otherwise, update the count and record it in freq_count.
            if new == 0:
                if num in id_count:
                    del id_count[num]
            else:
                id_count[num] = new
                freq_count[new] += 1
            
            # Update the running maximum frequency:
            # 1. If the new count for the current id is greater than the current max, update max_freq.
            if new > max_freq:
                max_freq = new
            # 2. If the id we updated was at the current maximum frequency (old == max_freq)
            #    and, after the update, no other id is left with that frequency,
            #    we need to adjust max_freq downward.
            elif old == max_freq and freq_count.get(max_freq, 0) == 0:
                # Search downward for the next highest frequency that exists.
                # Since counts are positive, the minimum value is 1.
                while max_freq > 0 and freq_count.get(max_freq, 0) == 0:
                    max_freq -= 1
            
            # Append the result for this step.
            # If the collection is empty, max_freq will be 0.
            ans.append(max_freq)
        
        return ans