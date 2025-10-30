class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        from collections import defaultdict
        
        # Dictionary to track the current frequency of each ID
        count_of_id = {}
        # Dictionary to track how many IDs have a given frequency
        freq_count = defaultdict(int)
        
        max_freq = 0
        ans = []
        
        for i in range(len(nums)):
            old_freq = count_of_id.get(nums[i], 0)
            new_freq = old_freq + freq[i]  # freq[i] can be positive (add) or negative (remove)
            
            # Decrease the count of IDs with the old frequency (if any)
            if old_freq > 0:
                freq_count[old_freq] -= 1
                if freq_count[old_freq] == 0:
                    del freq_count[old_freq]
            
            # Update the new frequency
            if new_freq == 0:
                # No occurrences left for this ID, remove it
                if nums[i] in count_of_id:
                    del count_of_id[nums[i]]
            else:
                # Set the new frequency
                count_of_id[nums[i]] = new_freq
                freq_count[new_freq] += 1
                # Update max_freq if needed
                if new_freq > max_freq:
                    max_freq = new_freq
            
            # If we've removed an ID that was at the max frequency,
            # we may need to lower max_freq
            while max_freq > 0 and max_freq not in freq_count:
                max_freq -= 1
            
            ans.append(max_freq)
        
        return ans