import collections
from typing import List

class Solution:
  """
  Solves the problem of finding the x-sum for sliding windows of size k.
  The x-sum calculation involves identifying the top x most frequent elements
  in a subarray (with tie-breaking based on element value) and summing their occurrences.
  """
  def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
    """
    Calculates the x-sum for all sliding windows of size k in the array nums.

    The x-sum calculation procedure for a subarray is:
    1. Count the occurrences of all elements in the subarray.
    2. Identify the top x most frequent elements. If two elements have the same 
       number of occurrences, the element with the bigger value is considered more frequent.
    3. Keep only the occurrences of these top x elements.
    4. Calculate the sum of the resulting array (sum of kept elements).
    Note: If a subarray has less than x distinct elements, its x-sum is the sum of the entire subarray.

    Args:
      nums: A list of integers. The input array.
      k: An integer. The size of the sliding window (subarray length).
      x: An integer. The number of top frequent elements to consider for the x-sum.

    Returns:
      A list of integers, answer, where answer[i] is the x-sum of the subarray nums[i:i+k].
      The length of the answer array is n - k + 1.
    """
    n = len(nums)
    answer = []  # Initialize the list to store the x-sum for each window

    # Iterate through all possible starting indices for subarrays of length k.
    # The loop runs from i = 0 up to n - k (inclusive).
    for i in range(n - k + 1):
        # Extract the current subarray (window) using slicing.
        # subarray contains elements from index i to i + k - 1.
        subarray = nums[i : i + k]

        # --- Calculate the x-sum for the current subarray ---

        # 1. Count frequencies of elements in the current subarray.
        # collections.Counter provides an efficient way to count hashable objects.
        counts = collections.Counter(subarray)
        # Convert the counter items into a list of (element, frequency) tuples.
        items = list(counts.items())

        # Get the number of distinct elements in the subarray.
        num_distinct = len(items)

        # Initialize the x-sum for the current window.
        current_x_sum = 0 

        # 2. Handle the special case as per the note:
        # If the subarray has less than x distinct elements, its x-sum is the sum of the subarray.
        if num_distinct < x:
            # Calculate the sum of all elements in the subarray.
            current_x_sum = sum(subarray)
        else:
            # 3. Identify the top x most frequent elements.
            # Sort the items list based on frequency and then element value.
            # The sorting key is a tuple: (-frequency, -element_value).
            # We use negative frequency (-item[1]) to sort in descending order of frequency.
            # We use negative element value (-item[0]) to sort in descending order of value
            # for elements with the same frequency (tie-breaking rule).
            # Python's default sort is stable, but using tuple comparison handles this directly.
            sorted_items = sorted(items, key=lambda item: (-item[1], -item[0]))

            # 4. Select the top x items (representing the top x frequency groups)
            # from the sorted list using slicing.
            top_x_items = sorted_items[:x]

            # 5. Calculate the x-sum by summing the contributions of these top x elements.
            # The contribution of each element is its value multiplied by its frequency.
            # This effectively sums all occurrences of the elements belonging to the top x groups.
            for element, frequency in top_x_items:
                current_x_sum += element * frequency

        # Add the computed x-sum for the current window to the results list.
        answer.append(current_x_sum)

    # Return the list containing the calculated x-sums for all sliding windows.
    return answer