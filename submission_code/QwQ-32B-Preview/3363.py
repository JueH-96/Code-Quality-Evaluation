from sortedcontainers import SortedDict
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        freq_map = {}
        freq_count = SortedDict()
        ans = []
        
        for i in range(len(nums)):
            if nums[i] in freq_map:
                # Remove the ID from its current frequency set
                current_f = freq_map[nums[i]]
                freq_count[current_f].remove(nums[i])
                if not freq_count[current_f]:
                    del freq_count[current_f]
                # Update the frequency
                freq_map[nums[i]] += freq[i]
                if freq_map[nums[i]] == 0:
                    del freq_map[nums[i]]
                else:
                    # Add to the new frequency set
                    new_f = freq_map[nums[i]]
                    if new_f not in freq_count:
                        freq_count[new_f] = set()
                    freq_count[new_f].add(nums[i])
            else:
                # Add the new frequency
                freq_map[nums[i]] = freq[i]
                if freq_map[nums[i]] == 0:
                    del freq_map[nums[i]]
                else:
                    # Add to the new frequency set
                    new_f = freq_map[nums[i]]
                    if new_f not in freq_count:
                        freq_count[new_f] = set()
                    freq_count[new_f].add(nums[i])
            
            # Get the current maximum frequency
            if freq_count:
                current_max = freq_count.peekitem(-1)[0]
            else:
                current_max = 0
            ans.append(current_max)
        
        return ans