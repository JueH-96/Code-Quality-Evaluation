import collections
from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []

        def calculate_x_sum(subarray: List[int], x: int) -> int:
            """Calculates the x-sum for a given subarray."""
            # 1. Count frequencies
            counts = collections.Counter(subarray)

            # 2. Handle case where distinct elements < x
            # If there are fewer distinct elements than x, sum all elements.
            distinct_elements_count = len(counts)
            if distinct_elements_count <= x:
                # As per the note, if distinct elements are less than or equal to x,
                # the x-sum is the sum of the original subarray.
                return sum(subarray)

            # 3. Find top x frequent elements with tie-breaking
            # Create a list of (frequency, value) tuples.
            # We want to sort descending by frequency first,
            # then descending by value for tie-breaking.
            # Sorting in Python is ascending by default for tuples.
            # We use negative values for frequency and value to achieve
            # descending sort order with Python's default sort.
            # The sort key will be (-frequency, -value).
            freq_items = []
            for val, freq in counts.items():
                freq_items.append((-freq, -val))

            # Sort the items. This sorts based on the tuple (-freq, -val)
            # Frequency descending, then value descending for ties.
            freq_items.sort()

            # The first x elements in the sorted list are the top x
            # Extract the original values of these top x elements
            # We only need the first x elements since distinct_elements_count > x.
            # len(freq_items) equals distinct_elements_count.
            top_x_vals = []
            for i in range(x):
                 top_x_vals.append(-freq_items[i][1]) # Convert -val back to val

            # 4. Calculate the sum of occurrences of top x elements
            # Sum = sum of (value * frequency) for each value in the top x list
            total_sum = 0
            for val in top_x_vals:
                total_sum += counts[val] * val

            return total_sum

        # Iterate through all sliding windows
        # The start index i ranges from 0 to n - k (inclusive)
        for i in range(n - k + 1):
            # Extract the current window (subarray) of size k
            current_window = nums[i : i + k]

            # Calculate the x-sum for the current window
            window_x_sum = calculate_x_sum(current_window, x)

            # Append the result to the answer list
            answer.append(window_x_sum)

        return answer