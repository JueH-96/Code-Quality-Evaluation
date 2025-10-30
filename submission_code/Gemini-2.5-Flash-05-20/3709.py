class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)

        # Iterate through all possible starting indices for a substring of length k.
        # A substring of length k starting at index `i` is s[i : i + k].
        # For this substring to be valid, `i + k` must be less than or equal to `n`.
        # So, the maximum value for `i` is `n - k`.
        # The `range(n - k + 1)` correctly generates indices from 0 to n - k (inclusive).
        for i in range(n - k + 1):
            # Extract the current substring of length k.
            current_substring = s[i : i + k]
            
            # Condition 1: The substring consists of only one distinct character.
            # We take the first character of the substring as the reference.
            char_in_substring = current_substring[0]
            
            # Check if all characters in the substring are the same as char_in_substring.
            is_uniform = True
            for char_in_sub in current_substring:
                if char_in_sub != char_in_substring:
                    is_uniform = False
                    break # Found a different character, so it's not uniform.
            
            # If the substring is not uniform, it doesn't meet the criteria.
            # Continue to the next possible substring.
            if not is_uniform:
                continue 

            # Condition 2: If there is a character immediately before the substring,
            # it must be different from the character in the substring.
            is_preceding_char_ok = True
            # A character immediately before exists if `i` is greater than 0.
            if i > 0: 
                if s[i - 1] == char_in_substring:
                    is_preceding_char_ok = False # The preceding character is the same.
            
            # If the preceding character condition is violated, move to the next substring.
            if not is_preceding_char_ok:
                continue

            # Condition 3: If there is a character immediately after the substring,
            # it must also be different from the character in the substring.
            is_following_char_ok = True
            # A character immediately after exists if `i + k` is less than `n`.
            # `s[i + k]` is the character right after the current substring.
            if i + k < n: 
                if s[i + k] == char_in_substring:
                    is_following_char_ok = False # The following character is the same.
            
            # If the following character condition is violated, move to the next substring.
            if not is_following_char_ok:
                continue

            # If all three conditions (uniformity, preceding character difference, 
            # and following character difference) are met, we have found a special substring.
            # Return True immediately as we only need to find one such instance.
            return True
        
        # If the loop completes without finding any such special substring after checking all possibilities,
        # return False.
        return False