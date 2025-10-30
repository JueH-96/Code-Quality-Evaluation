import collections
from typing import List

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        """
        Calculates the total frequencies of elements in nums that share the maximum frequency.

        Args:
            nums: A list of positive integers.

        Returns:
            The sum of frequencies for all elements that have the highest frequency.
        """

        # Step 1: Calculate the frequency of each element.
        # collections.Counter is efficient for this. It returns a dictionary-like object
        # where keys are the elements and values are their counts.
        frequencies = collections.Counter(nums)

        # Handle the edge case of an empty input list, although constraints state len >= 1.
        if not frequencies:
            return 0

        # Step 2: Find the maximum frequency among all elements.
        # We can get all the frequency values (counts) and find the maximum.
        max_frequency = 0
        for element in frequencies:
            if frequencies[element] > max_frequency:
                max_frequency = frequencies[element]
        
        # More concise way to find max frequency:
        # max_frequency = max(frequencies.values())

        # Step 3: Calculate the total frequency count for elements with the maximum frequency.
        # Iterate through the frequencies again. If an element's frequency matches
        # the maximum frequency, add that frequency to the total count.
        total_max_frequency_count = 0
        for element in frequencies:
            if frequencies[element] == max_frequency:
                # Add the frequency of this element (which is max_frequency)
                # to the total count.
                total_max_frequency_count += frequencies[element] # or simply += max_frequency

        # Alternative way for Step 3: Summing directly
        # total_max_frequency_count = sum(freq for freq in frequencies.values() if freq == max_frequency)

        return total_max_frequency_count