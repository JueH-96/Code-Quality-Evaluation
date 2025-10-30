from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # Count the number of even and odd elements.
        count_even = sum(1 for x in nums if x % 2 == 0)
        count_odd = len(nums) - count_even

        # For an alternating pattern starting with 'first'
        # When first and second parity are different, we try to pick as many numbers as possible
        # in order, switching the required parity on each selection.
        def alternating_length(first: int) -> int:
            expected = first
            length = 0
            for x in nums:
                if x % 2 == expected:
                    length += 1
                    # Toggle expected parity for the next position.
                    expected = 1 - expected
            return length

        # There are two cases for alternate sequences:
        # Pattern 1: starting with even, then odd, then even, ...
        alt1 = alternating_length(0)
        # Pattern 2: starting with odd, then even, then odd, ...
        alt2 = alternating_length(1)

        # Also, note that any subsequence where all elements have the same parity is valid.
        # So we also consider the maximum constant parity subsequences.
        # For all-evens, maximum valid subsequence has length = count_even.
        # For all-odds, maximum valid subsequence has length = count_odd.

        # The answer is the maximum found among these four possibilities.
        return max(count_even, count_odd, alt1, alt2)