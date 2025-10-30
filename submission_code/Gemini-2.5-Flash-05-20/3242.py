import collections
from typing import List

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Step 1: Calculate frequencies of all elements using collections.Counter.
        # This creates a dictionary-like object where keys are elements and values are their counts.
        # Example: nums = [1,2,2,3,1,4] -> freq_map = {1: 2, 2: 2, 3: 1, 4: 1}
        freq_map = collections.Counter(nums)

        # Step 2: Find the maximum frequency present in the array.
        # The problem constraints state that nums.length is at least 1,
        # so freq_map will never be empty.
        max_freq = 0
        # Get the maximum value among all frequencies.
        max_freq = max(freq_map.values())

        # Step 3 & 4: Calculate the total sum of frequencies for elements that have the maximum frequency.
        # We iterate through the frequency values obtained from freq_map.
        total_sum_of_max_frequencies = 0
        for freq in freq_map.values():
            # If the current frequency matches the maximum frequency, add it to our total sum.
            if freq == max_freq:
                total_sum_of_max_frequencies += freq
        
        # Return the accumulated total.
        return total_sum_of_max_frequencies