from collections import defaultdict
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        id_counts = defaultdict(int)
        max_freq = 0
        ans = []
        
        for i in range(len(nums)):
            id_counts[nums[i]] += freq[i]
            if freq[i] > 0:
                max_freq = max(max_freq, id_counts[nums[i]])
            else:
                # If the count of the current ID is now 0, remove it from the dictionary
                if id_counts[nums[i]] == 0:
                    del id_counts[nums[i]]
                # If we removed the most frequent ID, find the new max frequency
                if nums[i] in id_counts and id_counts[nums[i]] + freq[i] == max_freq:
                    max_freq = max(id_counts.values()) if id_counts else 0
            
            ans.append(max_freq)
        
        return ans