import collections
from typing import List

class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)

        # Step 1: Group indices by value
        # value_indices will store a dictionary where keys are the numbers
        # and values are sorted lists of indices where that number appears.
        # collections.defaultdict(list) is perfect for this as it initializes
        # a list for a key if it doesn't exist upon first access.
        value_indices = collections.defaultdict(list)
        for i, num in enumerate(nums):
            value_indices[num].append(i)
        
        # Initialize min_overall_seconds to a very large number.
        # We will update this with the minimum seconds found for any value.
        min_overall_seconds = float('inf')

        # Step 3: For each distinct value, calculate the time needed to make the entire
        # array equal to that value.
        for value, indices_list in value_indices.items():
            # If a value appears at all indices, it means the array is already
            # entirely composed of this value. So, 0 seconds are needed.
            # This is the absolute minimum, so we can return immediately.
            if len(indices_list) == n:
                return 0
            
            # current_max_gap will store the maximum "distance" between two consecutive
            # occurrences of the current 'value', considering both linear and circular gaps.
            # The time needed to fill such a gap is floor(current_max_gap / 2).
            current_max_gap = 0
            
            # Calculate maximum gap between consecutive occurrences in the linear sequence.
            # The indices_list is already sorted because we iterate through `nums` from 0 to n-1
            # when populating `value_indices`.
            for i in range(len(indices_list) - 1):
                # The gap is the difference between an index and the next.
                # Example: indices_list = [0, 4] in an array of size 5.
                # The gap 4 - 0 = 4 means elements at 1, 2, 3 need to be reached.
                gap = indices_list[i+1] - indices_list[i]
                current_max_gap = max(current_max_gap, gap)
            
            # Calculate the circular gap. This is the "wrap-around" distance from the
            # last occurrence of the value to the first occurrence.
            # Example: n=5, indices_list=[0, 4].
            # The gap between index 4 and index 0 (circularly) is:
            # (n - last_index) + first_index = (5 - 4) + 0 = 1.
            # This means nums[0] is one step away from nums[4] (or vice versa).
            # This correctly handles cases with a single occurrence too:
            # If indices_list = [p_0], then indices_list[-1] = p_0 and indices_list[0] = p_0.
            # The circular gap becomes n - p_0 + p_0 = n, which is correct for a single element.
            circular_gap = n - indices_list[-1] + indices_list[0]
            current_max_gap = max(current_max_gap, circular_gap)
            
            # The time required to make all elements this 'value' is determined by
            # the largest gap. An element in the middle of a gap of size 'G' needs
            # G // 2 seconds to be reached from either end.
            seconds_for_this_value = current_max_gap // 2
            
            # Update the overall minimum seconds needed.
            min_overall_seconds = min(min_overall_seconds, seconds_for_this_value)
            
        return min_overall_seconds