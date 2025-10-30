class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Find the position of the '*' wildcard in the pattern p.
        star_idx = p.find('*')
        
        # Split the pattern into prefix and suffix parts based on the '*'.
        prefix = p[:star_idx]
        suffix = p[star_idx+1:]

        # Get lengths for convenience.
        L_s = len(s)
        L_p_prefix = len(prefix)
        L_p_suffix = len(suffix)

        # Case 1: The pattern p is just "*".
        # A substring is defined as "a contiguous non-empty sequence of characters".
        # Since s.length >= 1 (constraint), s will always have at least one non-empty substring (e.g., s[0] or s itself).
        # If p is "*", the wildcard can match any non-empty substring in s.
        # Thus, a match is always possible.
        if L_p_prefix == 0 and L_p_suffix == 0:
            return True

        # Iterate over all possible starting positions (start_s) for a matching substring within s.
        # `start_s` is the inclusive beginning index of the potential substring.
        for start_s in range(L_s):
            # Check if the 'prefix' part of the pattern matches at the current `start_s`.
            # This check is only relevant if the prefix is not empty.
            if L_p_prefix > 0:
                # Ensure there are enough characters in s from `start_s` to accommodate the prefix.
                if start_s + L_p_prefix > L_s:
                    # Not enough space for the prefix, so this `start_s` cannot lead to a match.
                    continue 
                # Check if the substring of s matches the prefix.
                if s[start_s : start_s + L_p_prefix] != prefix:
                    # Prefix does not match, so this `start_s` cannot lead to a match.
                    continue
            
            # If we reach here, the prefix (or an empty prefix) has successfully matched at `start_s`.
            # Now, we need to find a suitable non-inclusive ending position (`end_s`) for the substring `s[start_s : end_s]`.

            # Calculate the minimum required `end_s`.
            # The substring must accommodate the prefix, the suffix, and the '*' (which can match an empty string).
            # This means the total length of the matched part (excluding the '*') must be at least `L_p_prefix + L_p_suffix`.
            # So, `end_s - start_s >= L_p_prefix + L_p_suffix`.
            # This implies `end_s >= start_s + L_p_prefix + L_p_suffix`.
            # Since `p != "*"` (handled above), `L_p_prefix + L_p_suffix` will always be greater than 0.
            # This ensures that `start_s + L_p_prefix + L_p_suffix` is always greater than `start_s`,
            # guaranteeing that any valid `s[start_s : end_s]` will be non-empty.
            min_valid_end_s = start_s + L_p_prefix + L_p_suffix
            
            # Iterate `end_s` from its minimum required value up to `len(s)`.
            # `end_s` is non-inclusive, so it can go up to `len(s)`.
            for end_s in range(min_valid_end_s, L_s + 1):
                # Check if the 'suffix' part of the pattern matches at the current `end_s`.
                # This check is only relevant if the suffix is not empty.
                if L_p_suffix > 0:
                    # Check if the substring of s ending at `end_s` matches the suffix.
                    # The slice `s[end_s - L_p_suffix : end_s]` represents the potential suffix part.
                    if s[end_s - L_p_suffix : end_s] != suffix:
                        # Suffix does not match, so this `end_s` is not valid for a match.
                        continue
                
                # If we reach this point, both the prefix (or empty prefix) starting at `start_s`
                # and the suffix (or empty suffix) ending at `end_s` have matched.
                # The characters between them (or an empty sequence) are covered by the `*`.
                # This means we have found a valid substring in `s` that matches the pattern `p`.
                return True

        # If the outer loops complete without finding any match, it means no such substring exists.
        return False