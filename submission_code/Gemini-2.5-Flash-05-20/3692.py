import bisect

class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        n = len(s)

        # Split the pattern p into three parts based on the two '*' characters.
        parts = p.split('*')
        prefix = parts[0]
        middle = parts[1]
        suffix = parts[2]

        L_p = len(prefix)
        L_m = len(middle)
        L_s = len(suffix)

        # Handle the special case where p is "**".
        # In this case, prefix, middle, and suffix are all empty strings.
        # The shortest matching substring is an empty string, with length 0.
        if L_p == 0 and L_m == 0 and L_s == 0:
            return 0
        
        # Calculate the minimum combined length of the non-wildcard parts.
        min_core_len_required = L_p + L_m + L_s
        
        # If the length of 's' is less than the minimum required length of the non-wildcard parts,
        # it's impossible to form a match.
        if n < min_core_len_required:
             return -1

        # KMP helper function to compute the Longest Prefix Suffix (LPS) array
        def compute_lps(pattern):
            length = 0  # length of the previous longest prefix suffix
            lps = [0] * len(pattern)
            i = 1
            while i < len(pattern):
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
            return lps

        # KMP helper function to find all occurrences of a pattern in text
        def find_all_matches_kmp(text, pattern):
            # This function is designed for non-empty patterns.
            # Empty pattern handling is done separately based on context (prefix, middle, suffix).
            if not pattern: 
                raise ValueError("find_all_matches_kmp expects non-empty pattern.")

            if len(pattern) > len(text):
                return []
            
            lps = compute_lps(pattern)
            matches = []
            i = 0  # index for text
            j = 0  # index for pattern

            while i < len(text):
                if pattern[j] == text[i]:
                    i += 1
                    j += 1
                
                if j == len(pattern):
                    # Found a match starting at index (i - j)
                    matches.append(i - j) 
                    # After a match, shift the pattern using the LPS array to find next potential match
                    j = lps[j - 1] 
                elif i < len(text) and pattern[j] != text[i]:
                    # Mismatch after some characters matched.
                    if j != 0:
                        # If a prefix of pattern was matched, use LPS to go to the next state.
                        j = lps[j - 1]
                    else:
                        # No prefix matched, move to the next character in text.
                        i += 1
            return matches

        # Precompute all possible starting indices for prefix, middle, and suffix using KMP.
        # Special handling for empty string patterns:
        # - An empty prefix always matches at the very beginning of the overall substring.
        #   So, `all_prefix_starts` only needs to contain `0`.
        all_prefix_starts = [0] if L_p == 0 else find_all_matches_kmp(s, prefix)
        
        # - An empty middle can match at any position from index 0 to `n` (inclusive).
        #   This is because it can represent an empty sequence between matched parts.
        all_middle_starts = list(range(n + 1)) if L_m == 0 else find_all_matches_kmp(s, middle)
        
        # - An empty suffix can match at any position from index 0 to `n` (inclusive).
        #   This defines the end point of the match; `j=n` means the suffix matches after `s[n-1]`.
        all_suffix_starts = list(range(n + 1)) if L_s == 0 else find_all_matches_kmp(s, suffix)

        min_total_len = float('inf')

        # Iterate through all possible starting positions of the `prefix` in `s`.
        for i in all_prefix_starts:
            # `mid_search_start_idx` is the earliest possible index in `s` where the `middle` part can start.
            # This is immediately after the current `prefix` match ends.
            mid_search_start_idx = i + L_p
            
            # Use binary search (bisect_left) to find the first occurrence of `middle`
            # that starts at or after `mid_search_start_idx`.
            mid_idx_in_list = bisect.bisect_left(all_middle_starts, mid_search_start_idx)
            
            # If no such `middle` match is found after the current `prefix` match,
            # then this `prefix` match cannot form a complete pattern match.
            # Move to the next `prefix` start.
            if mid_idx_in_list == len(all_middle_starts):
                continue 

            # `k` is the starting index of the earliest valid `middle` match.
            k = all_middle_starts[mid_idx_in_list]

            # `suffix_search_start_idx` is the earliest possible index in `s` where the `suffix` part can start.
            # This is immediately after the selected `middle` match ends.
            suffix_search_start_idx = k + L_m

            # Use binary search to find the first occurrence of `suffix`
            # that starts at or after `suffix_search_start_idx`.
            suffix_idx_in_list = bisect.bisect_left(all_suffix_starts, suffix_search_start_idx)
            
            # If no such `suffix` match is found after the current `middle` match,
            # then this sequence of `prefix` and `middle` cannot form a complete pattern match.
            # Move to the next `prefix` start.
            if suffix_idx_in_list == len(all_suffix_starts):
                continue

            # `j` is the starting index of the earliest valid `suffix` match.
            j = all_suffix_starts[suffix_idx_in_list]

            # Calculate the length of the current matched substring.
            # The substring starts at `i` (start of prefix) and ends at `j + L_s - 1` (end of suffix).
            # Length = (end_index + 1) - start_index = (j + L_s) - i.
            current_total_len = (j + L_s) - i

            # Update the minimum total length found so far.
            min_total_len = min(min_total_len, current_total_len)

        # If `min_total_len` is still `float('inf')`, it means no match was found.
        # Otherwise, return the shortest length found.
        return min_total_len if min_total_len != float('inf') else -1