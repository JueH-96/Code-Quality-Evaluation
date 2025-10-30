from typing import List
import collections

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        # Count the frequency of each element.
        # Example: [1, 2, 2, 3, 1, 4] -> Counter({1: 2, 2: 2, 3: 1, 4: 1})
        freq_map = collections.Counter(nums)
        
        # According to constraints, 1 <= nums.length, so nums is never empty,
        # and freq_map will have at least one entry.
        # If nums could be empty, we would handle it here, but it's not needed based on constraints.

        max_freq = 0
        total_frequency = 0

        # Iterate through the frequencies (the values in the counter).
        # Example: freq_map.values() might yield frequencies like [2, 2, 1, 1] for the example above.
        for freq in freq_map.values():
            if freq > max_freq:
                # Found a new maximum frequency.
                max_freq = freq
                # Reset total_frequency and start accumulating for this new max_freq.
                # The first element encountered with this new max_freq contributes 'freq' to the total.
                # Example: if we see freq=2 and max_freq was 0, max_freq becomes 2, total_frequency becomes 2.
                total_frequency = freq 
            elif freq == max_freq:
                # Found another element with the current maximum frequency.
                # Add its frequency to the total.
                # Example: if we see freq=2 and max_freq is already 2, total_frequency becomes current + 2.
                total_frequency += freq 

        # The final total_frequency is the sum of frequencies of all elements
        # that had the maximum frequency found.
        return total_frequency