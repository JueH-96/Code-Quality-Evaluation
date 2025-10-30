import collections # This import is not actually used and can be removed.

class Solution:
    def hasSpecialSubstring(self, s: str, k: int) -> bool:
        """
        Checks for a special substring of length k in s.

        A special substring satisfies these conditions:
        1. Length is exactly k.
        2. Consists of only one distinct character.
        3. If a character exists immediately before the substring, it must be different from the character in the substring.
        4. If a character exists immediately after the substring, it must also be different from the character in the substring.

        These conditions imply that the special substring must correspond precisely to a run
        (a consecutive sequence) of identical characters whose length is exactly k.
        If a run has length less than k, it doesn't meet condition 1.
        If a run has length greater than k, any substring of length k taken from within
        this run will violate either condition 3 (character before is the same) or
        condition 4 (character after is the same), or both.

        Therefore, we can solve this by finding runs of identical characters in s
        and checking if any run has a length exactly equal to k.

        Args:
            s: The input string. Constraints: 1 <= s.length <= 100, consists of lowercase English letters only.
            k: The required length of the substring. Constraints: 1 <= k <= s.length.

        Returns:
            True if such a special substring exists, False otherwise.
        """
        n = len(s)

        # Constraints guarantee 1 <= k <= n <= 100.
        # So, n is at least 1, and k is at least 1. No need for zero checks.

        # We iterate through the string, tracking the length of the current run
        # of identical characters.
        count = 0
        # This variable will store the character of the current run.
        # Its initial value doesn't matter as it's guaranteed to be set when i=0.
        current_char = '' 

        for i in range(n):
            if i == 0:
                # Initialize tracking for the first run starting at index 0.
                current_char = s[i]
                count = 1
            elif s[i] == current_char:
                # The character matches the current run's character, so extend the run.
                count += 1
            else:
                # The character s[i] is different from current_char.
                # This marks the end of the previous run (which ended at index i-1).
                # Check if the length of that just-completed run was exactly k.
                if count == k:
                    # If the completed run had length k, it satisfies all conditions
                    # for a special substring. The boundaries are handled because
                    # s[i] is different, and the character before the run (if any)
                    # must also have been different for this run to start where it did.
                    return True
                
                # Start tracking the new run beginning with the character s[i].
                current_char = s[i]
                count = 1 # Reset count for the new run.

        # After the loop finishes, we need to check the very last run of the string.
        # This run extends from the last character change (or index 0) up to index n-1.
        # The length of this last run is stored in 'count'.
        if count == k:
            # If the last run's length is exactly k, it forms a special substring.
            # The 'after' boundary condition is met because there's no character after.
            # The 'before' boundary condition is met due to how runs are defined.
            return True

        # If we have iterated through the entire string and checked the last run,
        # and found no run of length exactly k, then no such special substring exists.
        return False