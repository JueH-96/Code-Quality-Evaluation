from typing import List

class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        """
        Calculates the sum of good numbers in the array.

        An element nums[i] is considered good if it is strictly greater than
        the elements at indices i - k and i + k (if those indices exist).
        If neither of these indices exists, nums[i] is still considered good.

        Args:
            nums: A list of integers.
            k: An integer distance.

        Returns:
            The sum of all good elements in the array.
        """
        total_sum = 0
        n = len(nums)

        # Iterate through each element in the array
        for i in range(n):
            is_good = True

            # Check the condition against the left neighbor if it exists (i - k >= 0)
            left_idx = i - k
            if left_idx >= 0:
                # If the left neighbor exists, nums[i] must be strictly greater than nums[left_idx]
                if nums[i] <= nums[left_idx]:
                    is_good = False
                    # Optimization: If it fails the left check, no need to check the right.
                    # The structure `if is_good and right_idx < n:` handles this.

            # Check the condition against the right neighbor if it exists (i + k < n)
            right_idx = i + k
            # Only check the right neighbor if the element is still considered potentially good
            if is_good and right_idx < n:
                # If the right neighbor exists, nums[i] must be strictly greater than nums[right_idx]
                if nums[i] <= nums[right_idx]:
                    is_good = False

            # If is_good is still True, it means nums[i] satisfied the condition
            # for all existing neighbors (or had no neighbors to check against).
            # If neither neighbor exists (i - k < 0 AND i + k >= n), both if checks are skipped,
            # and is_good remains True, which is correct according to the problem statement.
            # (Based on the constraints 2 <= n <= 100 and 1 <= k <= floor(n / 2),
            # the case where neither index i-k nor i+k exists is impossible for any i,
            # but the logic correctly handles this rule as specified).

            if is_good:
                total_sum += nums[i]

        return total_sum