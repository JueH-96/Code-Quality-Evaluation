from typing import List
from collections import defaultdict

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        # Dictionary to keep track of the current frequency of each ID
        id_count = defaultdict(int)
        # Result array to store the most frequent ID count after each step
        ans = []
        # Variable to keep track of the current maximum frequency
        current_max_freq = 0
        
        for i in range(len(nums)):
            # Update the frequency of the current ID
            id_count[nums[i]] += freq[i]
            
            # If the frequency of the current ID is greater than the current maximum frequency
            if id_count[nums[i]] > current_max_freq:
                current_max_freq = id_count[nums[i]]
            
            # If the frequency of the current ID is less than the current maximum frequency
            # we need to check if the current maximum frequency is still valid
            elif id_count[nums[i]] < current_max_freq:
                # Check if the current maximum frequency is still valid
                if not any(count == current_max_freq for count in id_count.values()):
                    # If not, find the new maximum frequency
                    current_max_freq = max(id_count.values(), default=0)
            
            # Append the current maximum frequency to the result
            ans.append(current_max_freq)
        
        return ans