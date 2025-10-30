class Solution:
    def minChanges(self, s: str) -> int:
        """
        Calculates the minimum number of changes required to make a binary string beautiful.

        A beautiful string s can be partitioned into substrings of even length,
        each containing only 0's or only 1's.
        This is equivalent to the condition that for every even index i, s[i] must be
        equal to s[i+1] in the beautiful string.

        We iterate through the string in steps of 2, considering pairs of characters
        at indices (0, 1), (2, 3), ..., (n-2, n-1). For each pair s[i] and s[i+1],
        if they are different, we must make at least one change to make them equal
        in the beautiful string (either change s[i] to s[i+1] or s[i+1] to s[i]).
        The minimum changes for such a pair is 1. If they are already equal,
        no changes are needed for this pair (minimum changes is 0).
        The total minimum changes for the string is the sum of the minimum changes
        required for each pair independently.

        Args:
            s: A 0-indexed binary string of even length.

        Returns:
            The minimum number of changes needed to make s beautiful.
        """
        n = len(s)
        changes = 0
        # Iterate through the string taking steps of 2
        for i in range(0, n, 2):
            # Check if the pair of characters are different
            if s[i] != s[i+1]:
                # If they are different, we need one change for this pair
                changes += 1
        return changes