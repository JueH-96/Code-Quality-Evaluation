import collections
from typing import List

class Solution:
    def maxFrequencyElements(self, nums: List[int]) -> int:
        """
        Calculates the total frequencies of elements with the maximum frequency.
        """
        
        # Step 1: Count the frequency of each element.
        # e.g., for nums = [1,2,2,3,1,4], counts will be {1: 2, 2: 2, 3: 1, 4: 1}
        counts = collections.Counter(nums)
        
        # Step 2: Find the maximum frequency.
        # As per constraints, nums is non-empty, so counts.values() is safe.
        # For the example, max_frequency would be 2.
        max_frequency = max(counts.values())
        
        # Step 3: Calculate the total frequency.
        # This is the number of elements with max_frequency multiplied by max_frequency.
        
        # Count how many elements have the maximum frequency.
        # In the example, two elements (1 and 2) have the frequency of 2.
        elements_with_max_freq = 0
        for frequency in counts.values():
            if frequency == max_frequency:
                elements_with_max_freq += 1
                
        # The result is the number of such elements times their frequency.
        # 2 elements * frequency 2 = 4.
        return elements_with_max_freq * max_frequency