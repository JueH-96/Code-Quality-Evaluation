class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        from collections import defaultdict
        
        # freq_map: maps an ID -> its current frequency in the collection
        freq_map = defaultdict(int)
        
        # count_map: maps frequency -> how many IDs have this frequency
        count_map = defaultdict(int)
        
        max_freq = 0  # Tracks the maximum frequency among all IDs
        ans = []
        
        for i in range(len(nums)):
            id_val = nums[i]
            change = freq[i]
            
            # Current frequency of this id (if it exists, else 0)
            old_freq = freq_map[id_val]
            new_freq = old_freq + change
            
            # Update freq_map with new frequency
            freq_map[id_val] = new_freq
            
            # Decrease count of the old frequency
            if old_freq > 0:
                count_map[old_freq] -= 1
                if count_map[old_freq] == 0:
                    del count_map[old_freq]
            
            # Increase count of the new frequency if new_freq > 0
            if new_freq > 0:
                count_map[new_freq] += 1
            
            # Update max_freq
            if new_freq > max_freq:
                max_freq = new_freq
            else:
                # If the old frequency was the maximum and none left at that frequency
                # we may need to decrease max_freq.
                if old_freq == max_freq and old_freq not in count_map:
                    # Move max_freq down until we find a frequency that exists or 0
                    while max_freq > 0 and max_freq not in count_map:
                        max_freq -= 1
            
            # If the collection is empty, max_freq will be 0
            ans.append(max_freq if max_freq > 0 else 0)
        
        return ans