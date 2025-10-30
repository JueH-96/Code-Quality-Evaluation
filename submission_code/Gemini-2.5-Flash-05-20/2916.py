from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)

        # Base case: If the array has only one element, it's already considered
        # split into n (which is 1) non-empty arrays. No operations are needed.
        if n == 1:
            return True

        # Precompute prefix sums to calculate subarray sums efficiently in O(1) time.
        # prefix_sum[x] stores the sum of elements from nums[0] to nums[x-1].
        # prefix_sum[0] is 0.
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + nums[i]

        # dp[i][j] will be True if the subarray nums[i...j] can be fully split
        # into individual elements (i.e., n single-element arrays for a subarray of length n),
        # according to the problem's rules. Otherwise, it's False.
        dp = [[False] * n for _ in range(n)]

        # Iterate through subarray lengths, from 1 up to n.
        # This order ensures that when we compute dp[i][j], all required
        # dp values for smaller subarrays (dp[i][k] and dp[k+1][j]) are already known.
        for length in range(1, n + 1):
            # Iterate through all possible starting indices 'i' for a subarray of current 'length'.
            for i in range(n - length + 1):
                # Calculate the corresponding ending index 'j'.
                j = i + length - 1

                # Base case 1: Subarray of length 1 (e.g., [x]).
                # It's already a single element and cannot be split further.
                # It fulfills the condition of being an "array of length 1".
                if length == 1:
                    dp[i][j] = True
                    continue

                # Base case 2: Subarray of length 2 (e.g., [x, y]).
                # It must be split into [x] and [y].
                # Both [x] and [y] have length 1, satisfying the split condition.
                # Since they are length 1, they are also "fully split".
                # Thus, any length 2 subarray can always be fully split.
                if length == 2:
                    dp[i][j] = True
                    continue

                # For subarrays of length > 2:
                # Iterate through all possible split points 'k' within the current subarray nums[i...j].
                # A split point 'k' divides nums[i...j] into nums[i...k] (left part)
                # and nums[k+1...j] (right part).
                for k in range(i, j):
                    # Calculate properties (length and sum) of the left subarray.
                    left_len = k - i + 1
                    left_sum = prefix_sum[k+1] - prefix_sum[i]

                    # Calculate properties (length and sum) of the right subarray.
                    right_len = j - (k+1) + 1
                    right_sum = prefix_sum[j+1] - prefix_sum[k+1]

                    # Check if the immediate split (nums[i...j] -> nums[i...k], nums[k+1...j])
                    # is valid according to the problem's rules:
                    # (length is one OR sum is >= m) must hold for BOTH resulting subarrays.
                    left_part_valid_for_split = (left_len == 1 or left_sum >= m)
                    right_part_valid_for_split = (right_len == 1 or right_sum >= m)

                    # If the current split is valid AND both resulting subarrays can themselves
                    # be fully broken down into individual elements (recursively),
                    # then nums[i...j] can be fully split.
                    if left_part_valid_for_split and right_part_valid_for_split:
                        if dp[i][k] and dp[k+1][j]:
                            dp[i][j] = True
                            # Once we find a valid way to split, we don't need to check other 'k' values.
                            break
        
        # The final answer is whether the entire original array nums[0...n-1] can be fully split.
        return dp[0][n-1]