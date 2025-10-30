import collections # This import is not strictly needed for the O(1) space solution below, but good practice if using Counter potentially
from typing import List

class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        """
        Calculates the minimum length of the array nums after performing the removal operation any number of times.
        The operation involves choosing two indices i and j (i < j) such that nums[i] < nums[j], and removing both elements.
        The goal is to maximize the number of operations.

        Args:
            nums: A 0-indexed sorted array of integers in non-decreasing order.

        Returns:
            An integer denoting the minimum possible length of nums after removals.
        
        The core idea is based on the frequency of the most frequent element.
        Let N be the length of the array `nums`.
        Let `max_freq` be the frequency of the most frequent element in `nums`.

        Case 1: `max_freq > N / 2`.
        In this case, the most frequent element (let it be `m`) appears more times than all other elements combined.
        An operation requires removing two elements `x, y` such that `x < y`.
        If we remove a pair `(m, y)` where `m < y`, or `(x, m)` where `x < m`, we use one `m` and one non-`m` element.
        The number of non-`m` elements is `N - max_freq`.
        We can perform at most `N - max_freq` such operations involving `m`.
        After these operations, all `N - max_freq` non-`m` elements are removed, and `N - max_freq` copies of `m` are removed.
        The number of remaining elements is `max_freq - (N - max_freq) = 2 * max_freq - N`.
        All remaining elements are `m`, so no further operations are possible.
        The minimum length is `2 * max_freq - N`.

        Case 2: `max_freq <= N / 2`.
        In this case, no element appears more than half the times. It can be shown that we can pair up elements effectively.
        Consider pairing `nums[i]` with `nums[i + N // 2]` for `i` from `0` to `N // 2 - 1`.
        Since `max_freq <= N / 2`, it implies that `nums[i] < nums[i + N // 2]` for all valid `i`.
        This strategy allows performing `N // 2` operations.
        The number of elements removed is `2 * (N // 2)`.
        The remaining length is `N - 2 * (N // 2)`.
        If `N` is even, `N // 2 = N / 2`. Remaining length is `N - 2 * (N / 2) = N - N = 0`.
        If `N` is odd, `N // 2 = (N - 1) / 2`. Remaining length is `N - 2 * ((N - 1) / 2) = N - (N - 1) = 1`.
        This minimum length is `N % 2`.

        Thus, we need to calculate `max_freq` and apply the corresponding formula.
        """
        
        n = len(nums)
        if n == 0:
            return 0

        # Find the frequency of the most frequent element.
        # Since the array is sorted, we can do this efficiently in O(N) time and O(1) space.
        max_freq = 0
        # We need to correctly handle the first element and initialization
        if n > 0: # This check ensures nums[0] access is safe, though n=0 is already handled.
            current_val = nums[0]
            current_freq = 1
            max_freq = 1  # Initialize max_freq considering the first element
            
            for i in range(1, n):
                if nums[i] == current_val:
                    current_freq += 1
                else:
                    # Update max_freq when the element value changes
                    max_freq = max(max_freq, current_freq)
                    # Reset frequency counter for the new element
                    current_val = nums[i]
                    current_freq = 1
            
            # After the loop, update max_freq one last time for the last group of identical elements
            max_freq = max(max_freq, current_freq)

        # Determine the minimum length based on the comparison of max_freq with n/2
        if max_freq > n / 2:
            # Case 1: The most frequent element is the bottleneck.
            return 2 * max_freq - n
        else:
            # Case 2: No element is frequent enough to be a bottleneck.
            # Minimum length depends only on the parity of n.
            return n % 2