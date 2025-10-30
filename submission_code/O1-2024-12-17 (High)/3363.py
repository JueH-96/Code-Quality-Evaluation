class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        from collections import defaultdict
        
        # Dictionary to keep track of the current count/frequency of each ID
        current_count = defaultdict(int)
        
        # Dictionary to keep track of how many IDs have each particular frequency
        freq_count = defaultdict(int)
        
        max_freq = 0
        ans = []
        
        for i in range(len(nums)):
            val = nums[i]
            change = freq[i]
            
            # Old frequency of this ID
            old_count = current_count[val]
            # New frequency after this step
            new_count = old_count + change
            
            # Update current_count for this ID
            current_count[val] = new_count
            
            # If the old frequency is > 0, decrease its count in freq_count
            if old_count > 0:
                freq_count[old_count] -= 1
                if freq_count[old_count] == 0:
                    del freq_count[old_count]
            
            # If the new frequency is > 0, increase its count in freq_count
            if new_count > 0:
                freq_count[new_count] += 1
            
            # Update max_freq if we have a new maximum
            if new_count > max_freq:
                max_freq = new_count
            else:
                # If our old frequency was at max_freq and now there's no ID at that frequency
                while max_freq > 0 and max_freq not in freq_count:
                    max_freq -= 1
            
            # The collection might be empty, so answer could be 0
            ans.append(max_freq if max_freq > 0 else 0)
        
        return ans