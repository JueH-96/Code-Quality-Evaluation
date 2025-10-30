class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        from collections import defaultdict
        
        # Track count of each ID
        id_count = defaultdict(int)
        # Track how many IDs have each frequency count
        freq_count = defaultdict(int)
        # Result array
        ans = []
        # Track current maximum frequency
        max_freq = 0
        
        for i in range(len(nums)):
            id_val = nums[i]
            change = freq[i]
            
            # Get old count for this ID
            old_count = id_count[id_val]
            
            # Update to new count
            new_count = old_count + change
            id_count[id_val] = new_count
            
            # Update frequency tracking
            if old_count > 0:
                freq_count[old_count] -= 1
                if freq_count[old_count] == 0:
                    del freq_count[old_count]
            
            if new_count > 0:
                freq_count[new_count] += 1
            
            # Update max frequency
            if freq_count:
                max_freq = max(freq_count.keys())
            else:
                max_freq = 0
            
            ans.append(max_freq)
        
        return ans