class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        L = n + m - 1

        # Initialize word with empty strings (signifying not set by a 'T')
        word = [''] * L

        # Apply 'T' conditions and check for conflicts
        # For each i where str1[i] is 'T', word[i...i+m-1] must be str2
        for i in range(n):
            if str1[i] == 'T':
                # Iterate through the m characters of str2
                for k in range(m):
                    # The corresponding index in word is j = i + k
                    j = i + k
                    # Index j = i + k, where i in [0, n-1] and k in [0, m-1],
                    # ranges from 0+0=0 to (n-1)+(m-1) = n+m-2 = L-1.
                    # So j is always a valid index in word [0, L-1].

                    required_char = str2[k]
                    
                    # If word[j] is not set yet
                    if word[j] == '':
                        word[j] = required_char
                    # If word[j] is already set, it must match the required character
                    elif word[j] != required_char:
                        # Conflict detected: Different 'T' conditions impose different characters
                        return "" # No valid string can be generated

        # Precompute is_fixed array. An index j is fixed if its value was set by ANY 'T' condition.
        # A position is fixed if word[j] is not the initial empty string ''
        is_fixed = [word[j] != '' for j in range(L)]

        # Fill non-fixed positions (where word[j] is still '') with 'a'
        # This ensures the lexicographically smallest string satisfying 'T' conditions initially.
        for j in range(L):
            if word[j] == '':
                word[j] = 'a'

        # word is now the initial candidate string: satisfies all 'T' constraints and is lexicographically minimal elsewhere.

        # Check 'F' conditions and fix violations
        # Iterate through str1 indices i from 0 to n-1
        i = 0
        while i < n:
            # If str1[i] requires the substring word[i...i+m-1] to be NOT equal to str2
            if str1[i] == 'F':
                # Check if the current substring word[i : i + m] is equal to str2
                # list(str2) creates a list of characters from str2.
                # The slice word[i : i + m] creates a list of characters from word.
                # Comparing these two lists is O(M).
                # As analyzed, the slice word[i : i + m] is always valid and has length m for i < n.
                matches_str2 = (word[i : i + m] == list(str2))

                # If the substring matches str2, we have an 'F' violation
                if matches_str2:
                    # Violation detected: str1[i] == 'F' but word[i : i + m] == str2
                    # We need to modify word to break this equality.
                    # To maintain lexicographical minimality, find the latest possible index j in this window [i, i + m - 1]
                    # that we are allowed to change (i.e., not fixed by a 'T' condition).
                    fix_j = -1
                    # Iterate backward from the end of the window (index i + m - 1) to the start (index i)
                    # k goes from m-1 down to 0. word index j = i + k.
                    for k in range(m - 1, -1, -1):
                        j = i + k
                        # j is in [i, i+m-1], guaranteed to be in [0, L-1].
                        # If this position is NOT fixed by any 'T' condition
                        if not is_fixed[j]:
                            fix_j = j # This is the latest possible index we can modify
                            break # Found the latest, exit the k loop

                    # If no modifiable index was found in the window [i, i + m - 1]
                    if fix_j == -1:
                        # Cannot fix this violation without breaking a 'T' constraint.
                        # No valid string can be generated.
                        return ""

                    # We found the latest modifiable index fix_j in [i, i + m - 1].
                    # The character at word[fix_j] must be changed.
                    # Its current value is word[fix_j], which is equal to str2[fix_j - i] because the substring matched.
                    # Since is_fixed[fix_j] is False, word[fix_j] was initially 'a' (before any fixes).
                    # This implies str2[fix_j - i] must be 'a'.
                    # To make the smallest possible lexicographical change while breaking the match, we increment word[fix_j].
                    # The smallest character strictly greater than word[fix_j] is chr(ord(word[fix_j]) + 1).
                    # If word[fix_j] is 'z', we cannot increment it. If this happens at the latest modifiable position, no solution exists.
                    if word[fix_j] == 'z':
                         return "" # Cannot increment 'z'. No solution.

                    # Increment the character at the chosen fix_j index.
                    word[fix_j] = chr(ord(word[fix_j]) + 1)

                    # After incrementing word[fix_j], the string prefix up to fix_j is determined.
                    # To maintain overall lexicographical minimality, all unfixed characters
                    # at indices *after* fix_j (from fix_j + 1 up to L-1) should be set back to 'a'.
                    # This ensures that given the change at fix_j, the rest of the string is as small as possible.
                    for reset_j in range(fix_j + 1, L):
                        # reset_j is guaranteed to be < L by the range.
                        if not is_fixed[reset_j]:
                            word[reset_j] = 'a'

                    # The change at word[fix_j] might affect 'F' conditions at indices i' where
                    # word[i' : i' + m] includes word[fix_j]. This means i' is in [max(0, fix_j - m + 1), min(n - 1, fix_j)].
                    # We need to re-verify 'F' conditions starting from the earliest such index in str1,
                    # which is `max(0, fix_j - m + 1)`.
                    # We set the outer loop variable `i` such that the `i += 1` at the end of the loop
                    # makes the next iteration start checking from `max(0, fix_j - m + 1)`.
                    # Setting i to `max(0, fix_j - m + 1) - 1` achieves this.
                    # The max(-1, ...) handles the case where max(0, fix_j - m + 1) is 0.
                    i = max(-1, fix_j - m) # Equivalent to max(0, fix_j - m + 1) - 1

            # Move to the next index in str1. If a fix occurred, i was reset backward.
            # If no fix occurred (either str1[i]=='T' or str1[i]=='F' but no match), i increments normally.
            i += 1

        # If the while loop completes, all 'T' and 'F' conditions are satisfied.
        # The resulting word list represents the lexicographically smallest string.
        # Convert the list of characters into a string and return it.
        return "".join(word)