from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # A valid subsequence satisfies that (sub[i] + sub[i+1]) % 2 is constant for all valid i.
        # There are two possible values for this constant parity: 0 or 1.

        # Case 1: The sum of adjacent elements is always even (sum % 2 == 0).
        # This implies that adjacent elements in the subsequence must have the same parity.
        # Therefore, the entire subsequence must consist of elements of the same parity (either all odd or all even).
        # The length of the longest such subsequence is simply the count of the most frequent parity in the original array.
        count_even = 0
        count_odd = 0
        for num in nums:
            if num % 2 == 0:
                count_even += 1
            else:
                count_odd += 1
        max_len_same_parity = max(count_even, count_odd)

        # Case 2: The sum of adjacent elements is always odd (sum % 2 == 1).
        # This implies that adjacent elements in the subsequence must have different parities.
        # Therefore, the elements in the subsequence must have alternating parity (e.g., odd, even, odd, ... or even, odd, even, ...).
        # We can find the length of the longest such alternating subsequence using dynamic programming.
        # Let dp[0] be the maximum length of an alternating subsequence found so far that ends with an odd number.
        # Let dp[1] be the maximum length of an alternating subsequence found so far that ends with an even number.
        
        dp = [0, 0] # Initialize dp states. dp[0] for odd ending, dp[1] for even ending.

        for num in nums:
            if num % 2 == 1: # The current number is odd.
                # An alternating subsequence ending with this odd number must have the previous element being even.
                # The longest alternating subsequence ending with an even number found so far has length dp[1].
                # By appending the current odd number, we form a new alternating subsequence of length dp[1] + 1.
                # The current number itself also forms a valid subsequence of length 1.
                # The new maximum length for an alternating sequence ending in an odd number is the maximum of these possibilities.
                dp[0] = max(1, dp[1] + 1)
            else: # The current number is even.
                # An alternating subsequence ending with this even number must have the previous element being odd.
                # The longest alternating subsequence ending with an odd number found so far has length dp[0].
                # By appending the current even number, we form a new alternating subsequence of length dp[0] + 1.
                # The current number itself also forms a valid subsequence of length 1.
                # The new maximum length for an alternating sequence ending in an even number is the maximum of these possibilities.
                dp[1] = max(1, dp[0] + 1)

        # The maximum length of an alternating subsequence is the maximum of the lengths ending in odd or ending in even.
        max_len_alternating_parity = max(dp[0], dp[1])

        # The overall longest valid subsequence is the maximum of the lengths found in the two scenarios (all same parity or alternating parity).
        return max(max_len_same_parity, max_len_alternating_parity)