from typing import List

class Solution:
    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        """
        Determines if there exist two adjacent strictly increasing subarrays of length k.
        """
        n = len(nums)
        
        # The constraint `1 < 2 * k <= nums.length` guarantees that it's possible
        # to have two adjacent subarrays of length k.

        def is_strictly_increasing(start_idx: int) -> bool:
            """
            Checks if the subarray nums[start_idx : start_idx + k] is strictly increasing.
            
            This is done by ensuring that for every element in the subarray (except the last),
            it is smaller than the element that follows it.
            
            The `all()` function with a generator expression provides a concise and efficient
            way to perform this check. It short-circuits, stopping as soon as a condition
            is false. For k=1, the range is empty, and all() on an empty iterable correctly
            returns True.
            """
            return all(nums[i] < nums[i + 1] for i in range(start_idx, start_idx + k - 1))

        # We need to find a starting index 'a' for the first subarray.
        # The full block of two adjacent subarrays spans from index 'a' to 'a + 2*k - 1'.
        # To stay within the array bounds, 'a' can go up to 'n - 2*k'.
        for a in range(n - 2 * k + 1):
            # Check if the subarray starting at 'a' is strictly increasing.
            if is_strictly_increasing(a):
                # If the first is increasing, check the adjacent one starting at 'a + k'.
                if is_strictly_increasing(a + k):
                    # If both are increasing, we have found our pair.
                    return True
        
        # If the loop finishes without finding such a pair, none exist.
        return False