import bisect

class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:

        def find_all_occurrences(text, pattern):
            if not pattern:
                # An empty pattern matches before each character and after the last character
                return list(range(len(text) + 1))
            n = len(text)
            m = len(pattern)
            # If non-empty pattern is longer than text, no match possible
            if m > n:
                return []

            # KMP preprocessing (compute LPS array)
            lps = [0] * m
            length = 0 # length of the previous longest prefix suffix
            i = 1
            while i < m:
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1

            # KMP searching
            occurrences = []
            i = 0 # index for text
            j = 0 # index for pattern
            while i < n:
                if pattern[j] == text[i]:
                    i += 1
                    j += 1

                if j == m:
                    occurrences.append(i - j)
                    if j > 0: # Handle case where pattern is empty (j=0) - although already handled by the initial check
                       j = lps[j - 1]
                    else: # Should not happen if pattern is empty and we handle it before the loop
                       pass # Or break, as empty pattern matches all positions

                elif i < n and pattern[j] != text[i]:
                    if j != 0:
                        j = lps[j - 1]
                    else:
                        i += 1
            return occurrences

        # Find indices of '*' in p
        star_indices = [i for i, char in enumerate(p) if char == '*']
        star1_idx = star_indices[0]
        star2_idx = star_indices[1]

        # Split pattern into three parts
        part1 = p[:star1_idx]
        part2 = p[star1_idx + 1 : star2_idx]
        part3 = p[star2_idx + 1 :]

        # Find all occurrences of each part in s
        # starts1: start indices of part1 matches
        # starts2: start indices of part2 matches
        # starts3: start indices of part3 matches
        starts1 = find_all_occurrences(s, part1)
        starts2 = find_all_occurrences(s, part2)
        starts3 = find_all_occurrences(s, part3)

        len1 = len(part1)
        len2 = len(part2)
        len3 = len(part3)
        n = len(s) # Length of s

        min_len = float('inf')

        # If any non-empty part is not found, no match (empty parts are always found)
        if len1 > 0 and not starts1:
            return -1
        if len3 > 0 and not starts3:
             return -1
        # If part2 is non-empty but not found, no match
        if len2 > 0 and not starts2:
             return -1

        # Pointers for the sorted lists of occurrences
        # ptr_l points to the current candidate start index for part3 in starts3
        l_ptr = 0
        # ptr_k points to the current candidate start index for part2 in starts2
        # Initialize ptr_k to the first index k in starts2 such that k >= starts1[0] + len1
        # If starts1 is empty, the loop won't run, handled above.
        k_ptr = bisect.bisect_left(starts2, starts1[0] + len1) if starts1 else len(starts2)


        # Iterate through all possible start indices for the full substring (which is the start of part1)
        for i in starts1:
            # The segment after part1 starts at index i + len1
            after_part1_start = i + len1

            # Case 1: part2 is empty (len2 == 0). The pattern is part1 * part3.
            # A substring s[i:l+len3] matches if s[i:i+len1] == part1, s[l:l+len3] == part3, and i+len1 <= l.
            # We want to minimize (l + len3 - 1) - i + 1 = l + len3 - i.
            # For a fixed i, we need the smallest l in starts3 such that l >= i + len1.
            if len2 == 0:
                # Find the smallest l in starts3 such that l >= after_part1_start
                # Move l_ptr forward as long as the current l is too small
                while l_ptr < len(starts3) and starts3[l_ptr] < after_part1_start:
                    l_ptr += 1

                # If we found a valid l (start of part3)
                if l_ptr < len(starts3):
                    current_l = starts3[l_ptr]
                    # The length of the matching substring s[i : current_l + len3]
                    current_len = current_l + len3 - i
                    min_len = min(min_len, current_len)

            # Case 2: part2 is non-empty (len2 > 0). The pattern is part1 * part2 * part3.
            # A substring s[i:l+len3] matches if s[i:i+len1] == part1, s[l:l+len3] == part3, and
            # there exists k such that s[k:k+len2] == part2 and i + len1 <= k and k + len2 <= l.
            # Constraints: i \in starts1, k \in starts2, l \in starts3, i + len1 <= k, k + len2 <= l.
            # Minimize l + len3 - i.
            # For a fixed i, we need min { l + len3 | l \in starts3, \exists k \in starts2 s.t. i + len1 <= k <= l - len2 }.
            # This is min { l + len3 | l \in starts3, l >= i + len1, [i + len1, l - len2] intersects starts2 }.
            # The interval [i + len1, l - len2] intersects starts2 iff l - len2 >= k_first_ge_i_plus_len1 (if it exists).
            # Where k_first_ge_i_plus_len1 = min { k \in starts2 | k >= i + len1 }.
            # So, we need the smallest l in starts3 such that l >= max(i + len1, k_first_ge_i_plus_len1 + len2).

            else: # len2 > 0
                 # Find the first k in starts2 such that k >= after_part1_start.
                 # k_ptr points to the first k >= starts1[0] + len1. Advance it if needed for current i.
                while k_ptr < len(starts2) and starts2[k_ptr] < after_part1_start:
                    k_ptr += 1

                # If such a k exists
                if k_ptr < len(starts2):
                    k_first = starts2[k_ptr]
                    # Required l must be at least k_first + len2 to contain this part2 occurrence
                    required_l_from_k = k_first + len2
                    # Required l must also be at least i + len1 (part3 must start after part1 ends)
                    required_l_min_overall = max(after_part1_start, required_l_from_k)

                    # Find the smallest l in starts3 such that l >= required_l_min_overall
                    # l_ptr points to the first l >= starts3[0]. Advance it if needed based on required_l_min_overall.
                    while l_ptr < len(starts3) and starts3[l_ptr] < required_l_min_overall:
                         l_ptr += 1

                    # If we found a valid l (start of part3)
                    if l_ptr < len(starts3):
                         current_l = starts3[l_ptr]
                         # The length of the matching substring s[i : current_l + len3]
                         current_len = current_l + len3 - i
                         min_len = min(min_len, current_len)
                # else: No k in starts2 found after after_part1_start, so no match for this i (since len2 > 0)


        # Return min_len if a match was found, otherwise -1
        return min_len if min_len != float('inf') else -1