from typing import List
import bisect

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        # id_count maps each ID to its current frequency in the collection.
        id_count = {}
        # freq_count maps each frequency (greater than zero) to the number of IDs that have this frequency.
        freq_count = {}
        # active_freqs is a sorted list (ascending order) of all frequency values currently in use.
        active_freqs = []
        
        # Helper: add a new frequency value into active_freqs if it's not already present.
        def add_freq(val: int) -> None:
            idx = bisect.bisect_left(active_freqs, val)
            # Only add if it is not already there.
            if idx == len(active_freqs) or active_freqs[idx] != val:
                bisect.insort(active_freqs, val)
                
        # Helper: remove a frequency value from active_freqs.
        def remove_freq(val: int) -> None:
            idx = bisect.bisect_left(active_freqs, val)
            if idx < len(active_freqs) and active_freqs[idx] == val:
                active_freqs.pop(idx)
        
        res = []
        n = len(nums)
        for i in range(n):
            id_val = nums[i]
            change = freq[i]
            old_count = id_count.get(id_val, 0)
            new_count = old_count + change
            
            # If the ID already had a positive frequency, decrease its count in freq_count.
            if old_count > 0:
                freq_count[old_count] -= 1
                if freq_count[old_count] == 0:
                    del freq_count[old_count]
                    remove_freq(old_count)
            
            # If new_count > 0, update the frequency for the ID.
            if new_count > 0:
                id_count[id_val] = new_count
                if new_count in freq_count:
                    freq_count[new_count] += 1
                else:
                    freq_count[new_count] = 1
                    add_freq(new_count)
            else:
                # When new_count becomes 0, remove the ID from id_count.
                if id_val in id_count:
                    del id_count[id_val]
            
            # The most frequent ID's count is the maximum frequency value in the collection.
            if active_freqs:
                res.append(active_freqs[-1])
            else:
                res.append(0)
                
        return res