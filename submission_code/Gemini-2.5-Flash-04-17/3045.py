from typing import List

class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0

        descent_idx = -1
        descent_count = 0

        # Find descents (where nums[i] > nums[i+1]) for i from 0 to n-2
        for i in range(n - 1):
            if nums[i] > nums[i+1]:
                descent_count += 1
                descent_idx = i
                # If more than one descent within 0..n-2, it's impossible
                if descent_count > 1:
                    return -1

        # Case 1: No descents found in 0..n-2.
        # This means nums[0] < nums[1] < ... < nums[n-1] (due to distinct positive integers).
        # Array is already sorted.
        if descent_count == 0:
            return 0

        # Case 2: Exactly one descent found at index `descent_idx` (0 <= descent_idx < n-1).
        # This indicates a potential single rotation.
        # The array is split into two potentially increasing parts: [nums[0]...nums[descent_idx]] and [nums[descent_idx+1]...nums[n-1]].
        # For this to be a valid right-shifted sorted array, the second part must logically precede the first part in sorted order.
        # This requires the last element of the second part (nums[n-1]) to be less than the first element of the first part (nums[0]).
        if nums[n-1] < nums[0]:
            # The array is a valid right-shifted version of a sorted array.
            # The number of shifts required to sort it is the number of elements that were
            # originally at the end of the sorted array and were moved to the front.
            # The element nums[descent_idx + 1] is the smallest element and should be at index 0 after sorting.
            # It is currently at index `descent_idx + 1`.
            # To move an element from current index `j` to target index `p` using `k` right shifts, we need `(j + k) % n = p`.
            # We want to move the element currently at index `descent_idx + 1` to index 0.
            # So, `((descent_idx + 1) + k) % n = 0`.
            # This means `k % n = -(descent_idx + 1) % n`.
            # The smallest non-negative `k` is `(n - (descent_idx + 1)) % n`.
            # Since `descent_idx + 1` is between 1 and n-1, `n - (descent_idx + 1)` is between 1 and n-1.
            # So the number of shifts is `n - (descent_idx + 1)`.
            return n - (descent_idx + 1)
        else:
            # One descent exists, but the wrap-around condition fails (nums[n-1] >= nums[0]).
            # This means the array is not a valid single right-shift of a sorted array.
            return -1