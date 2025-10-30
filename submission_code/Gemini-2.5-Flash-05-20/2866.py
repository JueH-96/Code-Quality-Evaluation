from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_len = 0

        # Iterate through all possible starting points l
        for l in range(n):
            # Condition 1: nums[l] must be even
            # Condition 3 (part 1): nums[l] must be <= threshold
            if nums[l] % 2 == 0 and nums[l] <= threshold:
                # If nums[l] satisfies these initial conditions, it forms a valid subarray of length 1.
                # Initialize current_len to 1 and update max_len.
                current_len = 1
                max_len = max(max_len, current_len)

                # Try to extend this subarray to the right, starting from r = l + 1
                for r in range(l + 1, n):
                    # Condition 3 (part 2): nums[r] must be <= threshold
                    if nums[r] > threshold:
                        # If the current element exceeds the threshold, the subarray cannot be extended further
                        # while satisfying Condition 3. So, we break the inner loop.
                        break

                    # Condition 2: Parity must alternate.
                    # This means nums[r]'s parity must be different from nums[r-1]'s parity.
                    if (nums[r] % 2) == (nums[r-1] % 2):
                        # If parities are the same, the alternating condition is violated.
                        # The subarray cannot be extended further. So, we break the inner loop.
                        break

                    # If all conditions (threshold and alternating parity) are met for nums[r],
                    # the subarray [nums[l]...nums[r]] is valid.
                    current_len += 1
                    max_len = max(max_len, current_len)
            # If nums[l] does not satisfy the initial conditions (even and within threshold),
            # it cannot start a valid alternating subarray, so we simply move to the next l.
        
        return max_len