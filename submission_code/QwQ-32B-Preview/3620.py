from typing import List

class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        if not nums:
            return 0
        
        # Calculate the lower and upper bounds for each num
        lower_bounds = [num - k for num in nums]
        upper_bounds = [num + k for num in nums]
        
        # Sort the lower bounds
        lower_bounds_sorted = sorted(lower_bounds)
        
        # Initialize the first assigned value to the smallest lower bound
        current_value = lower_bounds_sorted[0]
        
        # Initialize the count of distinct elements
        distinct_count = 0
        
        # Iterate through each lower bound
        for lower, upper in zip(lower_bounds_sorted, sorted(upper_bounds)):
            # Assign the smallest possible unique value within the range
            assign_value = max(current_value, lower)
            # Ensure the assigned value does not exceed the upper bound
            if assign_value > upper:
                # If the assigned value exceeds the upper bound, skip this element
                continue
            # Assign the value and increment the count
            current_value = assign_value + 1
            distinct_count += 1
        
        return distinct_count