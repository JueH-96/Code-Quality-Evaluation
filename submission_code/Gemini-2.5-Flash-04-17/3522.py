from typing import List

class Solution:
    def resultsArray(self, nums: List[int], k: int) -> List[int]:
        """
        Calculates the power of all subarrays of size k.

        The power of an array is its maximum element if all its elements are
        consecutive and sorted in ascending order, and -1 otherwise.

        Args:
            nums: A list of integers.
            k: The size of the subarrays to consider.

        Returns:
            A list of integers representing the power of each subarray of size k.
        """
        n = len(nums)
        results = []

        # Helper function to check if a subarray is sorted and contains consecutive integers
        def is_consecutive_sorted(arr: List[int]) -> bool:
            """
            Checks if the given list is sorted in ascending order and if its
            elements are consecutive integers (difference between adjacent is 1).
            """
            # An array of size 1 or less is trivially consecutive and sorted
            if len(arr) <= 1:
                return True
            # Check if each element is exactly 1 greater than the previous one
            # This single check covers both sortedness (a[i+1] > a[i]) and
            # consecutiveness (a[i+1] - a[i] == 1)
            for i in range(len(arr) - 1):
                if arr[i+1] != arr[i] + 1:
                    return False
            # If the loop completes, the array is consecutive and sorted
            return True

        # Iterate through all possible starting indices for subarrays of size k
        # The possible starting indices are from 0 up to n - k
        for i in range(n - k + 1):
            # Extract the current subarray of size k using slicing
            sub_array = nums[i : i + k]

            # Check if the extracted subarray meets the conditions for a positive power
            if is_consecutive_sorted(sub_array):
                # If the subarray is consecutive and sorted, its maximum element
                # is its last element.
                results.append(sub_array[-1])
            else:
                # If the conditions are not met, the power is -1
                results.append(-1)

        # Return the list containing the calculated power for each subarray
        return results