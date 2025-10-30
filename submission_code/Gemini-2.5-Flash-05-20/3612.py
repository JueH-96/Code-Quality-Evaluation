from typing import List

class Solution:
    def _is_strictly_increasing(self, arr: List[int], start_idx: int, length: int) -> bool:
        """
        Helper function to check if a subarray of 'arr' starting at 'start_idx'
        and of 'length' is strictly increasing.

        A subarray of length 0 or 1 is considered strictly increasing by definition
        (as there are no pairs of adjacent elements to violate the condition).
        The loop logic naturally handles these cases: for length <= 1,
        the range for the loop will be empty, so it won't enter the loop body
        and will return True.
        """
        # This check ensures that the requested subarray does not go out of bounds.
        # In the context of the main function, this is implicitly handled by
        # the carefully chosen loop range for 'a', but it serves as a defensive check.
        if start_idx + length > len(arr):
            return False

        # Iterate from the first element of the subarray up to the second-to-last element.
        # We compare arr[i] with arr[i+1].
        # The loop finishes when i reaches (start_idx + length - 2),
        # so the last comparison is between arr[start_idx + length - 2] and arr[start_idx + length - 1].
        for i in range(start_idx, start_idx + length - 1):
            if arr[i] >= arr[i+1]:
                return False # Not strictly increasing (found a non-increasing pair)
        return True # All elements in the subarray were strictly increasing

    def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
        n = len(nums)

        # We need to find two adjacent subarrays of length k.
        # Let the first subarray start at index 'a'. It spans from `nums[a]` to `nums[a + k - 1]`.
        # The second adjacent subarray must then start at index `b = a + k`.
        # It spans from `nums[a + k]` to `nums[a + k + k - 1]`, which is `nums[a + k]` to `nums[a + 2k - 1]`.

        # For the second subarray to be entirely within the bounds of `nums`,
        # its last index `a + 2k - 1` must be less than `n`.
        # So, `a + 2k - 1 < n`
        # This simplifies to `a + 2k <= n`, or `a <= n - 2k`.
        # Therefore, the possible starting indices 'a' for the first subarray range
        # from `0` up to `n - 2k` (inclusive).
        # Python's `range(stop)` generates numbers up to `stop - 1`.
        # So, to include `n - 2k`, we use `range(n - 2k + 1)`.

        # The problem constraints `1 < 2 * k <= nums.length` guarantee that:
        # 1. `k` is at least 1 (since `2*k > 1`).
        # 2. There is always enough space for two adjacent subarrays of length `k` (`2*k <= n`).
        # This means `n - 2k` will always be non-negative, and `n - 2k + 1` will be at least 1,
        # ensuring the loop for 'a' runs at least once.

        for a in range(n - 2 * k + 1):
            # Check if the first subarray (starting at 'a' with length 'k') is strictly increasing.
            if self._is_strictly_increasing(nums, a, k):
                # If the first subarray is strictly increasing,
                # then check the second adjacent subarray.
                # Its starting index 'b' is 'a + k'.
                b = a + k
                if self._is_strictly_increasing(nums, b, k):
                    # Both subarrays are strictly increasing and are adjacent.
                    # We found a valid pair, so return True.
                    return True

        # If the loop completes without finding any such pair, return False.
        return False