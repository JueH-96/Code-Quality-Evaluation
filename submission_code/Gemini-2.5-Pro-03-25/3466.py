import collections
from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        """
        Counts the number of subarrays of nums where the bitwise AND of the elements equals k.

        The algorithm uses a dynamic programming approach based on the ending index of the subarrays.
        It leverages the property that the number of distinct bitwise AND values for all subarrays
        ending at a particular index `j` is small (at most logarithmic in the maximum value of nums).

        For each element `num` at index `j`, we maintain a dictionary `prev_ands` that stores the counts
        of different bitwise AND values for all subarrays ending at index `j-1`.
        We then compute `current_ands`, the dictionary for subarrays ending at index `j`, by:
        1. Considering the single-element subarray `[nums[j]]`.
        2. Extending all subarrays ending at `j-1` by including `nums[j]` and calculating the new AND values.

        The total count of subarrays with AND value `k` is updated incrementally as we process each element.

        Args:
            nums: A list of integers representing the input array.
            k: The target bitwise AND value for the subarrays.

        Returns:
            The total number of subarrays in `nums` whose bitwise AND equals `k`.
            Returns 0 if no such subarray exists or if the input array is empty (though constraints guarantee length >= 1).

        Time Complexity: O(N * log(max(nums))), where N is the length of `nums`. The number of distinct
                         AND values in `prev_ands` is at most O(log(max(nums))).
        Space Complexity: O(log(max(nums))) to store the `prev_ands` dictionary.
        """
        # total_count accumulates the number of valid subarrays found so far.
        total_count = 0

        # prev_ands stores {and_value: count} for subarrays ending at the *previous* index.
        # Keys are distinct bitwise AND values encountered.
        # Values are the counts of subarrays ending at the previous index having that specific AND value.
        # Using collections.defaultdict(int) initializes counts to 0 automatically for new keys.
        prev_ands = collections.defaultdict(int)

        # Iterate through each number `num` in the input array `nums`.
        # The loop implicitly iterates through the ending index `j` of the subarrays.
        for num in nums:
            # current_ands will store {and_value: count} for subarrays ending at the *current* index `j`.
            current_ands = collections.defaultdict(int)

            # --- Step 1: Process the single-element subarray [j, j] ---
            # This subarray consists only of the current number `num`.
            # Its bitwise AND value is just `num`.

            # Check if this single-element subarray itself satisfies the condition.
            if num == k:
                total_count += 1
            
            # Initialize the count for this single-element subarray in `current_ands`.
            # The AND value is `num`, and there is 1 such subarray ending here.
            current_ands[num] = 1

            # --- Step 2: Extend subarrays ending at the previous index j-1 ---
            # Iterate through each `(prev_and, count)` pair stored in `prev_ands`.
            # `prev_and` is the bitwise AND value of `count` subarrays ending at index `j-1`.
            for prev_and, count in prev_ands.items():
                # Calculate the bitwise AND value of the extended subarray ending at `j`.
                # If a subarray `nums[i..j-1]` had AND value `prev_and`, then the subarray
                # `nums[i..j]` (extended by `num = nums[j]`) has AND value `prev_and & num`.
                current_and = prev_and & num

                # Check if this new AND value equals the target `k`.
                # If `current_and == k`, it means that the `count` subarrays ending at `j-1`
                # (which had AND value `prev_and`), when extended to include `num`,
                # now form `count` valid subarrays ending at `j` with AND value `k`.
                # Add this `count` to the `total_count`.
                if current_and == k:
                    total_count += count

                # Update the `current_ands` dictionary for the current index `j`.
                # Add the `count` to the total count for the `current_and` value.
                # This aggregates counts for subarrays ending at `j` that result in the same `current_and` value.
                # Multiple `prev_and` values might map to the same `current_and`.
                current_ands[current_and] += count

            # After processing the current number `num`, the `current_ands` dictionary
            # holds the complete information about AND values for subarrays ending at index `j`.
            # Update `prev_ands` to be `current_ands` so it's ready for the next iteration (for index `j+1`).
            prev_ands = current_ands

        # After iterating through all numbers in `nums`, `total_count` contains the final answer.
        return total_count