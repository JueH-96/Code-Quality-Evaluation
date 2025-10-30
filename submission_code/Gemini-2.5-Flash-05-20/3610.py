import collections
from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []

        # Iterate through all possible subarrays (windows) of length k.
        # The window starts at index `i` and ends at `i + k - 1`.
        for i in range(n - k + 1):
            current_subarray = nums[i : i + k]
            
            # Calculate the x-sum for the current subarray using a helper method.
            current_x_sum = self._calculate_x_sum_for_subarray(current_subarray, x)
            answer.append(current_x_sum)
            
        return answer

    def _calculate_x_sum_for_subarray(self, sub_array: List[int], x: int) -> int:
        # Step 1: Count occurrences of all elements in the subarray.
        counts = collections.Counter(sub_array)
        
        # Step 2: Handle the special case:
        # If the number of distinct elements is less than x,
        # the x-sum is the sum of all elements in the subarray.
        if len(counts) < x:
            return sum(sub_array) 
            
        # Step 3: Identify the top x most frequent elements.
        # Create a list of (frequency, value) pairs from the counts.
        freq_value_pairs = []
        for val, freq in counts.items():
            freq_value_pairs.append((freq, val))
            
        # Sort the pairs based on the problem's criteria:
        # - Primary sort key: frequency (descending order, so use -freq)
        # - Secondary sort key (for tie-breaking): value (descending order, so use -val)
        # This ensures that if two elements have the same frequency, the one with the larger value is prioritized.
        freq_value_pairs.sort(key=lambda item: (-item[0], -item[1]))
        
        # Get the first 'x' elements from the sorted list.
        # These are the top x most frequent elements (considering tie-breaking).
        top_x_elements_info = freq_value_pairs[:x]
        
        # Step 4: Calculate the sum of the resulting array.
        # This involves summing the total occurrences of these top x elements.
        x_sum = 0
        for freq, val in top_x_elements_info:
            # For each top element 'val', add 'val' multiplied by its original frequency 'freq'
            x_sum += val * freq 
            
        return x_sum