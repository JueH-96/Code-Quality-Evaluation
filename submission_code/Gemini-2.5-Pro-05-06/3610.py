from typing import List

class Solution:
  def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
    n = len(nums)
    answer = []

    # Iterate through all possible starting positions of a subarray of length k
    # The number of such subarrays is n - k + 1
    for i in range(n - k + 1):
        # Current window (subarray) is nums[i ... i+k-1]
        
        # Step 1: Count occurrences of all elements in the current subarray.
        # Also, calculate the sum of all elements in the current subarray. This sum
        # will be used if the number of distinct elements is less than x.
        # Since nums[i] is constrained (1 to 50), a simple array for frequency counting is efficient.
        
        counts = [0] * 51  # counts[j] will store frequency of number j. Index 0 is unused.
        current_subarray_sum = 0
        
        # Iterate over the k elements of the current window
        for j_win_idx in range(k): # j_win_idx is index within the window [0, k-1]
            val = nums[i + j_win_idx]  # Element from the original nums array
            counts[val] += 1
            current_subarray_sum += val
        
        # Step 2: Prepare for sorting based on frequency and value.
        # Create a list of (frequency, value) pairs for elements that are actually present in the subarray.
        value_counts_pairs = []
        for val_candidate in range(1, 51):  # Iterate through all possible values (1 to 50)
            if counts[val_candidate] > 0:  # If this value appeared in the subarray
                value_counts_pairs.append((counts[val_candidate], val_candidate))
        
        # Sort these pairs to determine the "most frequent" elements.
        # The primary sort key is frequency (descending).
        # The secondary sort key is element value (descending) for tie-breaking.
        value_counts_pairs.sort(key=lambda p: (-p[0], -p[1]))
        
        # Step 3: Calculate the x-sum based on the sorted frequencies and the rules.
        num_distinct_elements = len(value_counts_pairs)
        
        current_x_sum = 0
        
        # The problem states a specific condition:
        # "Note that if an array has less than x distinct elements, 
        #  its x-sum is the sum of the array."
        if num_distinct_elements < x:
            current_x_sum = current_subarray_sum
        else:
            # Otherwise, the x-sum is calculated from the top x most frequent elements.
            # "Keep only the occurrences of the top x most frequent elements."
            # "Calculate the sum of the resulting array." (This means sum of these occurrences)
            
            # The `value_counts_pairs` list is already sorted according to frequency/value criteria.
            # We need to sum the contributions of the top x elements from this list.
            for top_item_idx in range(x): 
                # For each of these top x elements, get its original count (frequency) and its value.
                count_of_element, value_of_element = value_counts_pairs[top_item_idx]
                # Add all occurrences of this element to the x-sum.
                current_x_sum += count_of_element * value_of_element
        
        answer.append(current_x_sum)
            
    return answer