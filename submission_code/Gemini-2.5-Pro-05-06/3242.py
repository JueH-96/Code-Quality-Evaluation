import collections
from typing import List # Assuming List is from typing, as per standard practice

class Solution:
  def maxFrequencyElements(self, nums: List[int]) -> int:
    """
    Calculates the total frequencies of elements in nums that have the maximum frequency.
    """
    
    # Step 1: Count frequencies of each element.
    # nums constraint: 1 <= nums.length <= 100, so nums is not empty.
    # collections.Counter is efficient for this.
    # Example: nums = [1,2,2,3,1,4] -> freq_counts = Counter({1:2, 2:2, 3:1, 4:1})
    freq_counts = collections.Counter(nums)
    
    # Since nums.length >= 1, freq_counts is not empty, 
    # and therefore freq_counts.values() will also not be empty.
    # This ensures max() will not raise an error on an empty sequence.
    
    # Step 2: Find the maximum frequency among all elements.
    # Example: For Counter({1:2, 2:2, 3:1, 4:1}), its values are [2,2,1,1] (order may vary).
    # The maximum frequency is max([2,2,1,1]) = 2.
    max_frequency = max(freq_counts.values())
    
    # Step 3: Calculate the sum of frequencies for all elements that have this maximum frequency.
    # An element has the maximum frequency if its count in freq_counts is equal to max_frequency.
    # For each such element, its frequency (which is max_frequency) is added to the total.
    # Example: Elements 1 and 2 both have frequency 2 (the max_frequency).
    # Total = frequency_of_1 (which is 2) + frequency_of_2 (which is 2) = 4.
    
    total_sum_of_max_frequencies = 0
    # Iterate through the frequencies of the unique elements (the values in the Counter).
    for frequency_value in freq_counts.values():
        if frequency_value == max_frequency:
            # This element's frequency is the maximum. Add its frequency to the total.
            # Since frequency_value is equal to max_frequency here, this is equivalent to
            # adding max_frequency to the total.
            total_sum_of_max_frequencies += frequency_value 
            
    return total_sum_of_max_frequencies