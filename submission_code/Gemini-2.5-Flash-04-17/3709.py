class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        n = len(s)
        
        # Iterate through all possible start indices for a substring of length k.
        # A substring of length k starting at index i is s[i : i + k].
        # The valid start indices i are from 0 up to n - k.
        # The loop range range(n - k + 1) covers these indices.
        for i in range(n - k + 1):
            # The character that must be repeated in the potential uniform substring.
            # If the substring s[i : i + k] is uniform, all its characters must be the same as s[i].
            char = s[i] 
            
            # Check if the substring s[i : i + k] consists of only the character 'char'.
            # A simple way is to check if the substring equals 'char' repeated k times.
            if s[i : i + k] == char * k:
                # The substring is uniform. Now check the boundary conditions.
                
                # Check the character immediately before the substring.
                # This applies only if the substring is not at the beginning (i > 0).
                # The condition requires the character before (if it exists) to be DIFFERENT from 'char'.
                # This boolean is True if the preceding boundary condition is satisfied.
                # It's True if i == 0 (no preceding char, condition vacuously met) 
                # OR if i > 0 and s[i-1] is different from char.
                preceding_ok = (i == 0) or (s[i - 1] != char)
                    
                # Check the character immediately after the substring.
                # This applies only if the substring is not at the end (i + k < n).
                # The condition requires the character after (if it exists) to be DIFFERENT from 'char'.
                # This boolean is True if the succeeding boundary condition is satisfied.
                # It's True if i + k == n (no succeeding char, condition vacuously met) 
                # OR if i + k < n and s[i+k] is different from char.
                succeeding_ok = (i + k == n) or (s[i + k] != char)
                
                # If the substring is uniform AND satisfies both the preceding and succeeding boundary conditions, 
                # we have found a special substring. We can immediately return True.
                if preceding_ok and succeeding_ok:
                    return True
        
        # If the loop finishes without finding any such substring that satisfies all conditions,
        # it means no special substring exists in s. Return False.
        return False