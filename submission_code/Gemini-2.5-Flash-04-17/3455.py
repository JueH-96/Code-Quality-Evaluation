class Solution:
    def minimumLength(self, s: str) -> int:
        """
        Finds the minimum length of the string after applying the specified operation any number of times.

        The operation allows choosing an index i such that s[i] has a matching character
        to its left and right, and deleting the closest matching characters to the left and right.

        The problem can be solved using a two-pointer approach from the ends inwards.
        If the characters at the current left and right pointers s[l] and s[r] are the same,
        we can repeatedly apply the operation using internal characters of the same type
        as pivots, effectively removing the contiguous blocks of s[l] (or s[r]) from both ends.
        The optimal strategy is to remove the entire blocks of matching characters from
        the ends as long as the characters at the current ends match.

        Args:
            s: The input string.

        Returns:
            The minimum length of the final string.
        """
        l = 0
        r = len(s) - 1

        # Continue as long as the left pointer is before the right pointer
        # and the characters at the pointers are the same.
        while l < r and s[l] == s[r]:
            c = s[l]  # The character at the matching ends

            # Advance the left pointer past the contiguous block of character c
            # starting from the current left pointer position.
            # The loop condition l <= r ensures we don't go past the right pointer.
            while l <= r and s[l] == c:
                l += 1

            # Decrement the right pointer past the contiguous block of character c
            # ending at the current right pointer position.
            # The loop condition r >= l ensures we don't cross the left pointer's
            # new position.
            while r >= l and s[r] == c:
                r -= 1

        # After the loop finishes, either l >= r (meaning the entire string
        # was consumed by matching blocks), or s[l] != s[r] (meaning the
        # characters at the current ends do not match, and no further
        # operations on the ends are possible with this greedy strategy).
        # The length of the remaining substring is r - l + 1.
        # If l > r, the remaining substring is empty, and the length is 0.
        # max(0, r - l + 1) handles both cases correctly.
        return max(0, r - l + 1)