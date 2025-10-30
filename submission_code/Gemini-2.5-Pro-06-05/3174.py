class Solution:
    def minChanges(self, s: str) -> int:
        """
        Calculates the minimum number of changes to make a binary string beautiful.

        A string is beautiful if it can be partitioned into substrings of even
        length, where each substring is monochrome (all '0's or all '1's).

        This property implies that for a string to be beautiful, every pair of
        characters at indices (2k, 2k+1) must be identical. For example, a
        beautiful string like "001111" is composed of "00" | "11" | "11".

        Conversely, if all such pairs are identical, the string is beautiful.
        We can form a valid partition by grouping consecutive identical pairs
        (e.g., "00", "00" become "0000").

        Therefore, the task is to make s[2k] == s[2k+1] for all k.
        For each pair (s[2k], s[2k+1]):
        - If they are the same, 0 changes are needed for this pair.
        - If they are different ("01" or "10"), 1 change is needed to make them
          the same ("00" or "11").

        The total minimum changes is the count of pairs with different characters.
        """
        
        changes = 0
        # The string has an even length, so we can iterate in steps of 2.
        for i in range(0, len(s), 2):
            # Check the pair of characters at s[i] and s[i+1].
            if s[i] != s[i+1]:
                # If they are different, we must perform one change to make them identical.
                changes += 1
                
        return changes