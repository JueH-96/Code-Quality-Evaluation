class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        # Constraints: 1 <= k <= s.length <= 100

        # Iterate through all possible start indices of a substring of length k.
        # A substring of length k starting at index `i` runs from `s[i]` to `s[i+k-1]`.
        # The last possible starting index `i` is when the substring ends at the last character `s[n-1]`.
        # So, i + k - 1 = n - 1 => i = n - k.
        # The possible values for `i` are from 0 up to n - k (inclusive).
        for i in range(n - k + 1):
            # Get the potential character that the special substring should consist of.
            sub_char = s[i]

            # Check Condition 1: The substring s[i : i+k] consists of only `sub_char`.
            is_uniform = True
            # We check characters from the second one in the substring up to the end.
            # The range `range(1, k)` correctly handles k=1 (it's an empty range, so `is_uniform` remains True).
            for j in range(1, k):
                 # If any character in the substring (from the second one onwards) is different from the first character,
                 # it's not uniform.
                 if s[i + j] != sub_char:
                    is_uniform = False
                    break # No need to check further for this substring

            # If the current substring s[i : i+k] is indeed uniform (all characters are the same):
            if is_uniform:
                # Check Condition 3: The character immediately before the substring (if it exists) is different from `sub_char`.
                # A character immediately before exists only if the substring does not start at the very beginning of the string s (i.e., `i > 0`).
                # If `i == 0`, there is no character before, so this condition is considered satisfied.
                # The check `s[i - 1] != sub_char` is only performed if `i > 0` due to the short-circuiting nature of the 'or' operator.
                valid_before = (i == 0) or (s[i - 1] != sub_char)

                # Check Condition 4: The character immediately after the substring (if it exists) is different from `sub_char`.
                # A character immediately after exists only if the substring does not end at the very end of the string s (i.e., `i + k < n`).
                # The index of the character immediately after the substring is `i + k`.
                # If `i + k == n`, the substring ends exactly at the end of s, so there is no character after, and this condition is considered satisfied.
                # The check `s[i + k] != sub_char` is only performed if `i + k < n` due to the short-circuiting nature of the 'or' operator.
                valid_after = (i + k == n) or (s[i + k] != sub_char)

                # If the uniform substring also satisfies both boundary conditions (character before and character after, where applicable):
                if valid_before and valid_after:
                    # We have found at least one such substring. The problem asks if *there exists* such a substring.
                    return True # So we can immediately return True

        # If the loop finishes after checking all possible substrings of length k and none of them satisfied all the conditions:
        return False # No special substring was found.