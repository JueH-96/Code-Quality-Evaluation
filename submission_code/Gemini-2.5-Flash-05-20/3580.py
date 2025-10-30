class Solution:
    def minStartingIndex(self, s: str, pattern: str) -> int:
        n = len(s)
        m = len(pattern)

        # Basic check, though constraints guarantee pattern.length < s.length
        if m > n:
            return -1

        # Z-algorithm function to compute Z-array for a given text
        # Z[i] is the length of the longest common prefix between text and text[i:]
        def compute_z_array(text: str) -> list[int]:
            n_text = len(text)
            Z = [0] * n_text
            L, R = 0, 0  # [L, R] is the Z-box
            for i in range(1, n_text):
                if i <= R:
                    # If i is within the Z-box, use previous Z-values to initialize Z[i]
                    Z[i] = min(R - i + 1, Z[i - L])
                # Extend Z[i] by brute force comparison
                while i + Z[i] < n_text and text[Z[i]] == text[i + Z[i]]:
                    Z[i] += 1
                # If current Z-box extends beyond R, update L and R
                if i + Z[i] - 1 > R:
                    L, R = i, i + Z[i] - 1
            return Z

        # Step 1: Compute LCP (Longest Common Prefix) values
        # We construct a string: pattern + '#' + s (where '#' is a character not in s or pattern)
        # Z[len(pattern) + 1 + i] will give the LCP of 'pattern' with s[i:]
        text_fwd = pattern + '$' + s  # Using '$' as separator
        z_fwd = compute_z_array(text_fwd)
        
        lcp_values = [0] * (n - m + 1)
        for i in range(n - m + 1):
            lcp_values[i] = z_fwd[m + 1 + i] # Index for s[i:] in text_fwd

        # Step 2: Compute LCS (Longest Common Suffix) values
        # This is equivalent to LCP of reversed(pattern) with reversed(substring s[i:i+m])
        # We construct a string: reversed(pattern) + '#' + reversed(s)
        # The reversed substring s[i:i+m] corresponds to the slice s_rev[n-1-(i+m-1) : n-1-i+1]
        # which simplifies to s_rev[n-i-m : n-i].
        # The start index of this slice in s_rev is n-i-m.
        # So we query z_bwd at index len(pattern_rev) + 1 + (n-i-m).
        
        pattern_rev = pattern[::-1]
        s_rev = s[::-1]
        text_bwd = pattern_rev + '$' + s_rev # Using '$' as separator
        z_bwd = compute_z_array(text_bwd)
        
        lcs_values = [0] * (n - m + 1)
        for i in range(n - m + 1):
            # Calculate the starting index of the reversed substring s[i:i+m] within s_rev
            start_idx_in_s_rev = n - 1 - (i + m - 1) 
            lcs_values[i] = z_bwd[m + 1 + start_idx_in_s_rev]

        # Step 3: Iterate through possible starting indices in s and check the condition
        for i in range(n - m + 1):
            lcp = lcp_values[i]
            lcs = lcs_values[i]

            # num_matched_chars represents the total count of matched characters from prefix and suffix
            num_matched_chars = lcp + lcs

            # Case 1: 0 mismatches (perfect match)
            # If lcp + lcs covers at least 'm' characters, it implies a perfect match.
            # Example: if lcp = m, then num_matched_chars = m. If lcp = m-1 and lcs = 1, it's also m.
            if num_matched_chars >= m: 
                return i  # Found the smallest starting index

            # Case 2: 1 mismatch
            # If (lcp + lcs) equals m-1, it means exactly one character position is not covered by the matches.
            # This character is at s[i + lcp] in the substring and pattern[lcp] in the pattern.
            # We check if these two characters are indeed different.
            elif num_matched_chars == m - 1:
                # Ensure that lcp is within bounds for s[i + lcp] and pattern[lcp]
                # This is guaranteed by the logic: if lcp+lcs = m-1 and lcp < m, then lcp is a valid index.
                if s[i + lcp] != pattern[lcp]:
                    return i  # Found the smallest starting index with 1 mismatch
        
        # If no such substring is found after checking all possibilities
        return -1