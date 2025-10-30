import re # Using re module for potentially simpler solution, though manual loop is also valid.

class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        """
        Checks if the pattern p (with one '*') can be made a substring of s.

        The pattern p consists of a prefix, a wildcard '*', and a suffix.
        The '*' can match any sequence of zero or more characters.
        We need to find if there exists a substring in s that starts with the prefix,
        ends with the suffix, and has zero or more characters in between.

        Args:
            s: The main string (1 <= len(s) <= 50).
            p: The pattern string (1 <= len(p) <= 50, contains exactly one '*').

        Returns:
            True if such a substring exists in s, False otherwise.

        Example:
            s = "leetcode", p = "ee*e"
            prefix = "ee", suffix = "e"
            We look for a substring starting with "ee" and ending with "e".
            "eetcode" is a substring of "leetcode".
            "eetcode" starts with "ee", ends with "e". Match found -> True.

        Example:
            s = "luck", p = "u*"
            prefix = "u", suffix = ""
            We look for a substring starting with "u" and ending with "".
            "u", "uc", "uck" are substrings of "luck".
            "u" starts with "u", ends with "". Match found -> True.

        Example:
            s = "abc", p = "*c"
            prefix = "", suffix = "c"
            We look for a substring starting with "" and ending with "c".
            "abc", "bc", "c" are substrings of "abc".
            "abc" starts with "", ends with "c". Match found -> True.
        """
        
        # Find the index of the wildcard character '*'
        star_idx = p.find('*')
        # The problem statement guarantees that p contains exactly one '*',
        # so star_idx will always be a non-negative integer.

        # Extract the prefix (part before '*') and suffix (part after '*')
        prefix = p[:star_idx]
        suffix = p[star_idx + 1:]

        n = len(s)
        len_prefix = len(prefix)
        len_suffix = len(suffix)

        # --- Optimization ---
        # The minimum possible length of a substring matching the pattern p
        # is the length of the prefix plus the length of the suffix.
        # If the string s itself is shorter than this combined length,
        # it's impossible for any of its substrings to match the pattern.
        if n < len_prefix + len_suffix:
            return False

        # --- Main Logic: Iterative Search ---
        # We iterate through all possible starting positions 'i' for the prefix within s.
        # The prefix match would be the slice s[i : i + len_prefix].
        # The loop for 'i' must ensure that this slice is valid.
        # The last possible starting index for the prefix is n - len_prefix.
        # The range `range(n - len_prefix + 1)` covers indices from 0 to `n - len_prefix`.
        for i in range(n - len_prefix + 1):
            # Check if the substring of s starting at index 'i' matches the prefix.
            if s[i : i + len_prefix] == prefix:
                # If we found a match for the prefix, we now need to search for the suffix.
                # The suffix must start at an index 'j' such that:
                # 1. j >= i + len_prefix: The suffix must start at or after the position
                #    immediately following the end of the prefix match. This accounts for
                #    the '*' matching zero or more characters.
                # 2. The suffix match s[j : j + len_suffix] must fit within the bounds of s.
                #    This implies j + len_suffix <= n, or equivalently, j <= n - len_suffix.

                # Determine the valid range for the starting index 'j' of the suffix.
                # The earliest possible start index for the suffix.
                min_j = i + len_prefix
                # The latest possible start index for the suffix.
                max_j = n - len_suffix

                # Iterate through all valid starting indices 'j' for the suffix.
                # The range goes from min_j up to max_j (inclusive).
                for j in range(min_j, max_j + 1):
                    # Check if the substring of s starting at index 'j' matches the suffix.
                    if s[j : j + len_suffix] == suffix:
                        # If we find such a 'j', we have successfully found a substring in 's'
                        # (specifically s[i : j + len_suffix]) that matches the pattern 'p'.
                        # The prefix matches at 'i', the suffix matches at 'j',
                        # and the condition j >= i + len_prefix ensures the correct order
                        # and allows for the '*' wildcard match.
                        return True

        # If the outer loop finishes without finding any valid prefix-suffix combination
        # that satisfies the conditions, it means no substring of 's' matches the pattern 'p'.
        return False