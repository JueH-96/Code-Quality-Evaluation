from typing import List

class Solution:
    def longestAlternatingSubarray(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        max_length = 0
        i = 0  # Pointer for the potential start of a subarray

        # Iterate through the array to find potential starting points (l)
        while i < n:
            # Condition 1: The subarray must start with an even number
            # Condition 3 for the starting element: nums[i] must be <= threshold
            if nums[i] % 2 != 0 or nums[i] > threshold:
                # nums[i] cannot be the start of a valid subarray
                i += 1
                continue  # Move to the next element

            # If we reach here, nums[i] is a potential start (even and <= threshold).
            # This element itself forms a valid subarray of length 1.
            # We use 'j' as the pointer to extend the current subarray from index i.
            # The valid subarray starts at index i and ends at index j.
            j = i # Start checking from the current index itself.

            # Check if the single element subarray [nums[i]] is valid (length 1).
            # The conditions nums[i]%2==0 and nums[i]<=threshold are already checked above.
            # So, nums[i] itself is always a valid subarray of length 1 if we reach here.
            # We update max_length with at least 1 if a valid start is found.
            # This handles cases like [2, 2], threshold = 3, where the longest is 1.
            max_length = max(max_length, 1)

            # Try to extend the subarray from index i+1 onwards.
            j = i + 1
            while j < n:
                # Check Condition 3 for the current element nums[j]
                if nums[j] > threshold:
                    # Cannot extend further because nums[j] exceeds the threshold.
                    # The valid alternating subarray starting at i ends at j-1.
                    break

                # Check Condition 2: Adjacent elements must have different parity.
                # Compare nums[j] with the previous element nums[j-1].
                # j-1 is always a valid index (>= i) since j starts from i+1.
                if nums[j] % 2 == nums[j-1] % 2:
                    # The alternating property is broken.
                    # The valid alternating subarray starting at i ends at j-1.
                    break

                # If both checks passed, nums[j] can be included in the current subarray.
                # The subarray nums[i...j] is valid. Its length is j - i + 1.
                # Update the maximum length found so far.
                max_length = max(max_length, j - i + 1)

                # Move to the next element to potentially extend further.
                j += 1

            # The inner while loop (extending with j) finished.
            # The longest alternating subarray starting at 'i' ends at index j-1.
            # The element at index 'j' (if j < n) caused the break (either threshold or parity).
            # The next potential start for a *new* alternating subarray must be at or after index 'j'.
            # This is because any starting index between i+1 and j-1 would eventually encounter
            # the same element nums[j] and fail the same condition, or fail earlier.
            # We advance the outer loop pointer 'i' to 'j'.
            i = j

        return max_length