from typing import List

class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        n = len(nums)
        memo = {}

        def dp(i, prev_parity):
            if i == n:
                return 0
            if (i, prev_parity) in memo:
                return memo[(i, prev_parity)]

            # Calculate the current parity
            curr_parity = nums[i] % 2

            # Calculate the penalty if we move to the current position
            penalty = x if prev_parity != -1 and prev_parity != curr_parity else 0

            # Calculate the score if we move to the current position
            score = nums[i] - penalty + dp(i + 1, curr_parity)

            # Calculate the score if we skip the current position
            skip_score = dp(i + 1, prev_parity)

            # Store the maximum score in the memo
            memo[(i, prev_parity)] = max(score, skip_score)

            return memo[(i, prev_parity)]

        # Start the dynamic programming from the first position
        return nums[0] + dp(1, nums[0] % 2)