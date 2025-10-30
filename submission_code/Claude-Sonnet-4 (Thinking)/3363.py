class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        count = {}  # ID -> count
        freq_count = {}  # frequency -> number of IDs with this frequency
        result = []
        
        for id_val, change in zip(nums, freq):
            # Get current count
            old_count = count.get(id_val, 0)
            new_count = old_count + change
            
            # Update frequency count for old count
            if old_count > 0:
                freq_count[old_count] -= 1
                if freq_count[old_count] == 0:
                    del freq_count[old_count]
            
            # Update count and frequency count for new count
            if new_count > 0:
                count[id_val] = new_count
                freq_count[new_count] = freq_count.get(new_count, 0) + 1
            elif new_count == 0:
                # Remove ID from count if it becomes 0
                if id_val in count:
                    del count[id_val]
            
            # Find max frequency
            max_freq = max(freq_count.keys()) if freq_count else 0
            result.append(max_freq)
        
        return result