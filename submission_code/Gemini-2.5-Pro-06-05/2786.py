class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        
        # According to constraints, 1 <= n <= 50.
        # Any single character is a valid semi-repetitive substring.
        # The minimum possible answer for a non-empty string is 1.
        max_len = 1
        
        # `left` is the starting index of the current sliding window.
        left = 0
        
        # `last_pair_pos` stores the starting index of the most recent pair.
        # e.g., if s[i] == s[i-1], we store i-1.
        # A value of -1 indicates no pair has been seen yet.
        last_pair_pos = -1

        # We slide the window's right edge from the second character to the end.
        for right in range(1, n):
            # Check if the new character forms a pair with the previous one.
            if s[right] == s[right-1]:
                # A pair is found at indices (right-1, right).
                
                # If `last_pair_pos` is not -1, it means we have already encountered
                # a pair. This new pair is the second one, which violates the
                # semi-repetitive condition for the window s[left...right].
                if last_pair_pos != -1:
                    # To make the window valid again, we must shrink it from the left.
                    # The new window must start right after the first pair.
                    # The first pair started at `last_pair_pos`, so we set the new `left`
                    # to `last_pair_pos + 1`.
                    left = last_pair_pos + 1
                
                # Update `last_pair_pos` to the starting index of the newly found pair.
                # This pair is now the single allowed pair in the current valid window s[left...right].
                last_pair_pos = right - 1

            # The window s[left...right] is now guaranteed to be semi-repetitive.
            # We update `max_len` with the length of this current valid window.
            max_len = max(max_len, right - left + 1)

        return max_len