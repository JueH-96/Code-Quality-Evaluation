class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        """
        Finds the minimum possible sum of a k-avoiding array of length n.

        A k-avoiding array is an array of distinct positive integers
        such that there does not exist any pair of distinct elements that sum to k.

        To minimize the sum, we select the smallest possible distinct positive integers
        that satisfy the k-avoiding property. We iterate through positive integers
        1, 2, 3, ... and greedily pick a number if it has not been forbidden by
        a previously selected number.

        An integer `i` is forbidden if `k - i` has been picked, where `k - i`
        is a positive integer and `k - i != i`.

        Args:
            n: The desired length of the array.
            k: The sum to avoid for any pair of distinct elements.

        Returns:
            The minimum possible sum of a k-avoiding array of length n.
        """
        selected_count = 0
        current_sum = 0
        forbidden = set()  # Set to store numbers that cannot be picked
        current_num = 1    # The positive integer currently being considered

        # We need to select `n` distinct positive integers
        while selected_count < n:
            # If the current number is not forbidden, we can pick it
            if current_num not in forbidden:
                current_sum += current_num
                selected_count += 1

                # If we pick `current_num`, its complement `k - current_num`
                # cannot be picked later if it's a positive integer distinct
                # from `current_num`.
                complement = k - current_num
                
                # Condition: complement must be positive AND distinct from current_num
                if complement > 0 and complement != current_num:
                    forbidden.add(complement)

            # Move to the next positive integer
            current_num += 1

        return current_sum