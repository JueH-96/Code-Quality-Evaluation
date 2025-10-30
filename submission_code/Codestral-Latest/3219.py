from typing import List
import heapq

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        # Create a list of tuples (value, index) to keep track of original indices
        indexed_nums = [(num, i) for i, num in enumerate(nums)]

        # Sort the list based on the values
        indexed_nums.sort()

        # Initialize the result array with None
        result = [None] * len(nums)

        # Use a min-heap to keep track of the current window of indices
        window = []

        for i, (num, idx) in enumerate(indexed_nums):
            # Add the current element to the window
            heapq.heappush(window, (idx, num))

            # If the difference between the largest and smallest element in the window
            # exceeds the limit, remove the smallest element from the window
            while window[-1][1] - window[0][1] > limit:
                heapq.heappop(window)

            # The smallest element in the window should be placed at the current index
            result[window[0][0]] = num

        return result