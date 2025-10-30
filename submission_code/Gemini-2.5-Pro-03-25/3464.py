import math
from typing import List

class Solution:
    """
    Solves the problem of finding the maximum total cost by splitting an array `nums` into subarrays.
    The cost of a subarray nums[l..r] is defined as nums[l] - nums[l+1] + ... + nums[r] * (-1)^(r-l).
    The total cost is the sum of costs of all subarrays in the partition.

    The approach uses dynamic programming with O(N) time and O(1) space complexity.
    At each index `i`, we maintain two values:
    1. `pos_prev`: The maximum total cost considering the prefix `nums[0..i-1]`, where the last element `nums[i-1]` contributed positively (i.e., `+nums[i-1]`). This implies `nums[i-1]` was at an odd position (1st, 3rd, ...) within its subarray.
    2. `neg_prev`: The maximum total cost considering the prefix `nums[0..i-1]`, where the last element `nums[i-1]` contributed negatively (i.e., `-nums[i-1]`). This implies `nums[i-1]` was at an even position (2nd, 4th, ...) within its subarray.

    We iterate through the array from the second element (`i=1`) and calculate the current maximum costs (`current_pos`, `current_neg`) for the prefix ending at index `i` based on the values from index `i-1` (`pos_prev`, `neg_prev`).

    Recurrence relations:
    - `current_pos`: Represents the maximum cost ending at index `i` with `nums[i]` having a positive contribution (`+nums[i]`). This can happen in two ways:
        a) Start a new subarray at index `i`. `nums[i]` is the 1st element (odd position). The cost is the maximum cost achieved up to index `i-1` (which is `max(pos_prev, neg_prev)`) plus `nums[i]`.
        b) Extend a subarray ending at `i-1` where `nums[i-1]` had a negative contribution (`neg_prev`). Since `nums[i-1]` was at an even position, `nums[i]` will be at an odd position. The cost is `neg_prev + nums[i]`.
        Since `neg_prev <= max(pos_prev, neg_prev)`, the cost from case (a) is always greater than or equal to the cost from case (b). Thus, `current_pos = max(pos_prev, neg_prev) + nums[i]`.
    - `current_neg`: Represents the maximum cost ending at index `i` with `nums[i]` having a negative contribution (`-nums[i]`). This can only happen by extending a subarray ending at `i-1` where `nums[i-1]` had a positive contribution (`pos_prev`). Since `nums[i-1]` was at an odd position, `nums[i]` will be at an even position. The cost `pos_prev` already includes the `+nums[i-1]` term. When `nums[i]` is appended with a negative sign, the total cost becomes `pos_prev - nums[i]`. (The logic is that the cost associated with the path ending before `nums[i-1]` plus `nums[i-1] - nums[i]` results in this total). Thus, `current_neg = pos_prev - nums[i]`.

    Base case: For the first element `nums[0]` (index 0):
    - `pos_prev = nums[0]`: `nums[0]` must start a subarray, so it's at position 1 (odd), contributing positively.
    - `neg_prev = -math.inf`: It's impossible for `nums[0]` to be at an even position.

    The final answer is the maximum of the two states after processing the last element `nums[n-1]`.
    """
    def maximumTotalCost(self, nums: List[int]) -> int:
        """
        Calculates the maximum total cost achievable by splitting the array nums into subarrays.

        Args:
            nums: A list of integers. The length is guaranteed to be at least 1.

        Returns:
            An integer representing the maximum total cost.
        """
        n = len(nums)
        
        # Initialize DP states based on the first element nums[0].
        # pos_prev holds the max cost ending at index i-1 with nums[i-1] having positive contribution.
        pos_prev = nums[0] 
        # neg_prev holds the max cost ending at index i-1 with nums[i-1] having negative contribution.
        # Initialize neg_prev to negative infinity because nums[0] cannot have negative contribution.
        neg_prev = -math.inf 
        
        # Iterate from the second element (index 1) to the end of the array.
        for i in range(1, n):
            # Calculate `current_neg`: Max cost ending at index `i` with `nums[i]` having negative contribution.
            # This state is reached by extending a state ending at `i-1` where `nums[i-1]` was positive.
            current_neg = pos_prev - nums[i]
            
            # Calculate `current_pos`: Max cost ending at index `i` with `nums[i]` having positive contribution.
            # This state is primarily reached by starting a new subarray at `i`.
            # The cost is the max cost achieved up to `i-1` plus `nums[i]`.
            current_pos = max(pos_prev, neg_prev) + nums[i]

            # Update the previous states for the next iteration.
            pos_prev = current_pos
            neg_prev = current_neg
            
        # After iterating through all elements, the final maximum total cost is the maximum of the 
        # two possible states for the last element `nums[n-1]`.
        return max(pos_prev, neg_prev)