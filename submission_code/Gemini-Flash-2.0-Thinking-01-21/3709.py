class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        
        # Iterate through all possible starting indices for a substring of length k
        # A substring of length k starting at index i goes from s[i] to s[i+k-1].
        # The starting index i can range from 0 up to n - k.
        # If i = n - k, the substring is s[n-k : n-k+k] = s[n-k : n], which ends at index n-1.
        # The range goes up to n - k + 1 (exclusive), so it includes n - k.
        for i in range(n - k + 1):
            # Check condition 1: The substring s[i:i+k] consists of only one distinct character.
            # We can check this by verifying that all characters in the substring
            # are the same as the first character s[i].
            char = s[i]
            is_single_char = True
            # Check characters from index i+1 up to i+k-1
            # The range for j is 1, 2, ..., k-1.
            # The indices checked are i+1, i+2, ..., i+k-1.
            # If k=1, range(1, 1) is empty, so the loop is skipped, and is_single_char remains True, which is correct.
            for j in range(1, k):
                 if s[i+j] != char:
                     is_single_char = False
                     break # Found a different character, so this is not a single-character substring

            # If the substring consists of only one distinct character
            if is_single_char:
                # Now check the boundary conditions (conditions 2 and 3)
                
                # Condition 2: If there is a character immediately before, it must be different.
                # This condition applies only if the substring does not start at the beginning of s (i.e., i > 0).
                before_condition_met = True
                if i > 0: # Check if there is a character at index i-1
                    if s[i - 1] == char:
                        # The character before is the same, condition 2 is NOT met
                        before_condition_met = False 
                
                # Condition 3: If there is a character immediately after, it must be different.
                # This condition applies only if the substring does not end at the end of s (i.e., i + k < n).
                # The index immediately after the substring s[i:i+k] is i + k.
                after_condition_met = True
                if i + k < n: # Check if the index immediately after (i+k) is within the bounds of s
                    if s[i + k] == char:
                         # The character after is the same, condition 3 is NOT met
                        after_condition_met = False
                
                # If all conditions (single character, before different if exists, after different if exists) are met,
                # we have found the special substring.
                if before_condition_met and after_condition_met:
                    return True # Found a special substring satisfying all conditions

        # If the loop finishes without finding any substring that satisfies all conditions
        return False