from collections import defaultdict
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        id_counts = defaultdict(int)
        freq_counts = defaultdict(int)
        current_max = 0
        ans = []
        
        for i in range(len(nums)):
            num = nums[i]
            delta = freq[i]
            
            old_count = id_counts[num]
            if old_count > 0:
                freq_counts[old_count] -= 1
                if freq_counts[old_count] == 0:
                    del freq_counts[old_count]
            
            new_count = old_count + delta
            id_counts[num] = new_count
            
            if new_count > 0:
                freq_counts[new_count] += 1
            
            # Update current_max
            if new_count > current_max:
                current_max = new_count
            elif old_count == current_max:
                if freq_counts.get(old_count, 0) == 0:
                    if not freq_counts:
                        current_max = 0
                    else:
                        current_max = max(freq_counts.keys())
            
            ans.append(current_max)
        
        return ans