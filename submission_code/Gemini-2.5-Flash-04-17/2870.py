from typing import List

class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        """
        Finds the maximum length of an alternating subarray.

        An alternating subarray s of length m is called alternating if:
        1. m is greater than 1.
        2. s_1 = s_0 + 1.
        3. The 0-indexed subarray s looks like [s_0, s_1, s_0, s_1,...].
           This implies s[k] = s[k-2] for k >= 2.

        Args:
            nums: A 0-indexed integer array.

        Returns:
            The maximum length of all alternating subarrays or -1 if none exists.
        """
        max_len = -1
        n = len(nums)

        # Iterate through all possible starting indices i of an alternating subarray.
        # An alternating subarray must have length > 1, so it must contain at least nums[i] and nums[i+1].
        # The first element nums[i] is s_0. The second element nums[i+1] is s_1.
        # The condition s_1 = s_0 + 1 must be met for nums[i] and nums[i+1].
        # So, i can range from 0 up to n-2.
        for i in range(n - 1):
            # Check if the first two elements form the base case of an alternating subarray: s_1 = s_0 + 1.
            if nums[i+1] - nums[i] == 1:
                # Found an alternating subarray of length 2 starting at i: [nums[i], nums[i+1]].
                current_len = 2
                # Update max_len if this is the longest one found so far.
                # We only consider length > 1, and current_len is already 2.
                max_len = max(max_len, current_len)

                # Now, try to extend this subarray using the rule s[k] = s[k-2] for k >= 2.
                # The subarray indices start from 0 (s_0 at nums[i], s_1 at nums[i+1], s_2 at nums[i+2], etc.).
                # The rule s[k] = s[k-2] for k >= 2 translates to checking elements from index k=2 onwards in the subarray.
                # The element at subarray index k corresponds to index i + k in the original array nums.
                # The rule s[k] = s[k-2] means nums[i+k] should be equal to nums[i+k-2].
                # We iterate through potential subsequent elements in nums starting from index j = i + 2.
                # The index j in nums corresponds to the subarray index k = j - i.
                # The rule becomes nums[i + (j-i)] == nums[i + (j-i) - 2], which simplifies to nums[j] == nums[j-2].
                for j in range(i + 2, n):
                    # Check if the current element nums[j] follows the s[k] = s[k-2] pattern.
                    # The required pattern is nums[j] == nums[j-2].
                    if nums[j] == nums[j-2]:
                        # The pattern holds. The alternating subarray starting at i extends to include nums[j].
                        # Its new length is (j - i + 1).
                        # Since `current_len` was tracking the length up to the previous valid element (j-1),
                        # adding nums[j] increases the length by 1.
                        current_len += 1
                        # Update max_len if this longer subarray is the maximum found so far.
                        max_len = max(max_len, current_len)
                    else:
                        # The pattern s[k] = s[k-2] is broken at index j.
                        # The alternating subarray starting at index i cannot be extended further than index j-1.
                        # Break out of this inner loop and proceed to the next potential starting index i.
                        break # Pattern broken for this starting index i.

        return max_len