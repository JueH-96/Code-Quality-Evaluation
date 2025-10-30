from typing import List
from collections import defaultdict

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        """
        Counts the number of subarrays where the bitwise AND of elements equals k.

        Args:
            nums: A list of integers.
            k: The target bitwise AND value.

        Returns:
            The number of subarrays with a bitwise AND of k.
        """
        # prev_and_map stores {and_value: count} for distinct AND values found among
        # subarrays ending at the *previous* index (j-1).
        # For example, if at index j-1, prev_and_map[v] = c, it means there are c
        # subarrays ending at j-1 whose bitwise AND is v.
        # Initialize with an empty map, as there are no subarrays ending before index 0.
        prev_and_map = {}

        total_count = 0

        # Iterate through each number in the input array, using its index `j` as the
        # right endpoint of potential subarrays. `num` is nums[j].
        for num in nums:
            # current_and_map stores {and_value: count} for distinct AND values found among
            # subarrays ending at the *current* index (j).
            # Initialize for the current index j.
            current_and_map = defaultdict(int)

            # 1. Extend the subarrays that ended at the previous index (j-1) by including the current number (nums[j]).
            # For each AND value `prev_val` achieved by `count` subarrays ending at j-1 (from prev_and_map),
            # extending them with `num` results in a new AND value `prev_val & num`.
            # These `count` subarrays now end at index j and have the new AND value.
            for prev_val, count in prev_and_map.items():
                current_and_map[prev_val & num] += count

            # 2. Add the new subarray consisting only of the current number `num` itself.
            # This subarray starts and ends at the current index j.
            # Its AND value is simply `num`. There is one such subarray.
            current_and_map[num] += 1

            # 3. After processing all subarrays ending at the current index `j`,
            # the number of these subarrays whose bitwise AND equals k is given by current_and_map[k].
            # We add this count to our running total.
            # defaultdict(int) provides a default value of 0 if k is not a key in current_and_map,
            # so accessing current_and_map[k] is safe even if no subarray ending at j has AND value k.
            total_count += current_and_map[k]

            # 4. The current_and_map represents the state for the current index j.
            # For the next iteration (index j+1), this map becomes the "previous" state.
            # We discard the old prev_and_map.
            prev_and_map = current_and_map

        # After iterating through all possible right endpoints, total_count holds the sum
        # of counts for each endpoint, which is the total number of subarrays
        # across the entire array whose bitwise AND equals k.
        return total_count