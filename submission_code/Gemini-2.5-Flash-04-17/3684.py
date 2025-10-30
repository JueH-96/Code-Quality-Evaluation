class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Find the index of the single '*' character in the pattern p.
        star_index = p.find('*')

        # Extract the prefix and suffix parts of the pattern based on the '*' position.
        # p_prefix is the part of p before the '*'.
        # p_suffix is the part of p after the '*'.
        p_prefix = p[:star_index]
        p_suffix = p[star_index + 1:]

        # Calculate the minimum length a matching substring in s must have.
        # This is the combined length of the prefix and the suffix.
        # The '*' can match an empty sequence, so the minimum length occurs when '*' matches nothing.
        min_len = len(p_prefix) + len(p_suffix)

        # Edge case: If the pattern p is just '*', it can match any non-empty substring of s.
        # Since the problem constraints guarantee 1 <= s.length, s always has at least one non-empty substring.
        # If min_len is 0, it means p_prefix and p_suffix are both empty, so p was just '*'.
        if min_len == 0:
            return True

        # If the minimum required length for a match is greater than the length of s,
        # it's impossible for any substring of s to be long enough.
        if min_len > len(s):
            return False

        # Iterate through all possible start indices 'i' for a potential matching substring in s.
        # A substring starting at 'i' must be long enough to accommodate the prefix.
        # The latest 'i' can be is such that i + len(p_prefix) <= len(s), i.e., i <= len(s) - len(p_prefix).
        for i in range(len(s) - len(p_prefix) + 1):

            # Check if the substring of s starting at 'i' matches the prefix part of the pattern.
            if s[i : i + len(p_prefix)] == p_prefix:

                # If the prefix matches starting at index 'i', we now need to find if there's an end index 'j'
                # for the same substring s[i:j+1] such that it also matches the suffix.
                # The substring s[i:j+1] must have a length of at least min_len.
                # This means j - i + 1 >= min_len, which simplifies to j >= i + min_len - 1.
                # The substring must end within s, so the maximum value for j is len(s) - 1.
                # So, 'j' can range from i + min_len - 1 up to len(s) - 1.
                for j in range(i + min_len - 1, len(s)):

                    # Check if the substring of s ending at 'j' matches the suffix part of the pattern.
                    # The suffix part within s[i:j+1] starts at index (j+1) - len(p_suffix) = j - len(p_suffix) + 1.
                    # It ends at index j.
                    suffix_start_index_in_s = j - len(p_suffix) + 1

                    # Note: The loop range for 'j' (from i + min_len - 1) already ensures that
                    # suffix_start_index_in_s >= i + len(p_prefix), meaning the suffix does not
                    # start before the prefix ends. This is because:
                    # j >= i + len(p_prefix) + len(p_suffix) - 1
                    # j - len(p_suffix) + 1 >= i + len(p_prefix)
                    # suffix_start_index_in_s >= i + len(p_prefix)

                    if s[suffix_start_index_in_s : j + 1] == p_suffix:
                        # If both prefix and suffix match for the substring s[i:j+1],
                        # then the pattern p can be made a substring of s.
                        return True # Found a match, we can stop and return True.

        # If the loops complete without finding any matching substring,
        # it means no substring of s matches the pattern p.
        return False