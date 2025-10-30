class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # A substring is defined as non-empty.
        # If p is just "*", it can be replaced by any sequence. To be a non-empty
        # substring, we can replace it with any non-empty substring of s.
        # Since s is guaranteed to be non-empty (1 <= s.length), such a
        # substring always exists (e.g., s[0]).
        if p == "*":
            return True

        # The pattern p has exactly one '*', so we can split it into a prefix
        # and a suffix.
        star_idx = p.find('*')
        prefix = p[:star_idx]
        suffix = p[star_idx + 1:]
        
        len_s = len(s)
        len_prefix = len(prefix)

        # We are looking for a substring in s that starts with `prefix` and,
        # at some point after the prefix, contains `suffix`.
        
        # Iterate through all possible starting positions of the prefix in s.
        for i in range(len_s - len_prefix + 1):
            # Check if s has the prefix starting at index i.
            if s.startswith(prefix, i):
                # If a prefix match is found, we search for the suffix.
                # The suffix must start at or after the end of the prefix.
                # The position right after the prefix match is `i + len_prefix`.
                search_start = i + len_prefix
                
                # s.find() searches for the first occurrence of `suffix` at or after
                # `search_start`. It returns -1 if not found. This also correctly
                # handles an empty suffix (it would be "found" at `search_start`).
                if s.find(suffix, search_start) != -1:
                    # A valid prefix and suffix were found in the correct order.
                    # This implies a matching substring exists.
                    return True
                    
        # If the loop completes without finding any such arrangement, no match exists.
        return False