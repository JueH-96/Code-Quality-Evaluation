import math
from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        """
        Calculates the maximum length of an alternating subarray in nums using a single pass O(N) approach.

        An alternating subarray s of length m satisfies:
        1. m > 1
        2. s[1] = s[0] + 1
        3. s[k] - s[k-1] = 1 if k is odd (position in the subarray)
        4. s[k] - s[k-1] = -1 if k is even (k >= 2)

        Args:
          nums: The input list of integers. Constraints: 2 <= len(nums) <= 100, 1 <= nums[i] <= 10^4.

        Returns:
          The maximum length of an alternating subarray (which must be > 1), or -1 if none exists.
        """
        n = len(nums)
        # Initialize max_len to 0. If no alternating subarray of length >= 2 is found,
        # we will return -1 at the end.
        max_len = 0  
        # current_len stores the length of the alternating subarray ending at the *previous* index (i-1).
        # A value of 0 indicates no active alternating subarray ending at the previous position.
        # We modify it to store the length of the alternating subarray ending at the *current* index i within the loop.
        current_len = 0 

        for i in range(1, n):
            diff = nums[i] - nums[i-1]

            # Check if we can extend the alternating subarray ending at index i-1
            if current_len > 0:
                # Determine the expected difference for the current step (nums[i] - nums[i-1]).
                # The previous subarray had length `current_len`. The current element `nums[i]`
                # would be at index `current_len` (0-based) within the subarray.
                # The difference `nums[i] - nums[i-1]` corresponds to `s[k] - s[k-1]` where `k = current_len`.
                # Expected diff = 1 if k (current_len) is odd.
                # Expected diff = -1 if k (current_len) is even.
                expected_diff = 1 if (current_len % 2 != 0) else -1

                if diff == expected_diff:
                    # The alternating pattern continues, increment the length.
                    current_len += 1
                else:
                    # The pattern is broken.
                    # Check if the current pair (nums[i-1], nums[i]) can start a *new*
                    # alternating subarray. This requires the difference to be 1.
                    if diff == 1:
                        # Start a new alternating subarray of length 2.
                        current_len = 2
                    else:
                        # Cannot start a new one, reset the current length.
                        current_len = 0
            else:
                # If we were not in an alternating subarray ending at index i-1 (current_len was 0).
                # Check if the current pair (nums[i-1], nums[i]) can start a new alternating subarray.
                # This requires the difference to be 1.
                if diff == 1:
                    # Start a new alternating subarray of length 2.
                    current_len = 2
                else:
                    # Cannot start a subarray with this pair, current length remains 0.
                    current_len = 0 

            # After processing the pair ending at index i and potentially updating current_len,
            # update the overall maximum length found so far.
            # current_len now represents the length of the alternating subarray ending at index i.
            if current_len > 0: 
                 max_len = max(max_len, current_len)

        # If max_len is still 0 after checking all pairs, it means no alternating subarray 
        # of length >= 2 was found. The problem requires returning -1 in this case.
        # Otherwise, return the maximum length found.
        return max_len if max_len > 0 else -1