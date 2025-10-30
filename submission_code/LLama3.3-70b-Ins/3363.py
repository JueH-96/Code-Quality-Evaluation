from typing import List
from collections import defaultdict

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        # Initialize a dictionary to store the frequency of each ID
        id_freq = defaultdict(int)
        
        # Initialize a list to store the result
        ans = []
        
        # Iterate over the IDs and their frequencies
        for num, f in zip(nums, freq):
            # Update the frequency of the current ID
            id_freq[num] += f
            
            # If the frequency of the current ID is negative, set it to 0
            if id_freq[num] < 0:
                id_freq[num] = 0
            
            # Find the maximum frequency
            max_freq = max(id_freq.values()) if id_freq else 0
            
            # Append the maximum frequency to the result
            ans.append(max_freq)
        
        return ans