class Solution:
    def possibleStringCount(self, word: str) -> int:
        """
        Calculates the number of possible original strings Alice might have typed.

        Alice might have pressed a key for too long at most once, resulting in
        a block of consecutive identical characters in the final string 'word'
        being longer than in the original string.

        Args:
            word: The final string displayed on Alice's screen.

        Returns:
            The total number of possible original strings.
        """
        # The count starts at 1 to include the case where no character was extended.
        # If the final string 'word' was the intended string, this is one possibility.
        count = 1

        n = len(word)
        # Constraint guarantees n >= 1, but handle 0 defensively if needed elsewhere.
        # if n == 0:
        #     return 0

        i = 0
        while i < n:
            current_char = word[i]
            block_length = 0
            j = i
            # Find the length of the current block of identical characters.
            while j < n and word[j] == current_char:
                block_length += 1
                j += 1

            # If a block has length m > 1 in the final string, it could have
            # resulted from an original block of the same character with length k,
            # where 1 <= k < m.
            # The possible original lengths are 1, 2, ..., m-1.
            # There are (m - 1) such possible original lengths.
            # Each of these corresponds to a unique original string where this
            # block was the *only* one extended.
            # For example, if a block 'ccc' (length 3) appears in 'word', it could
            # have come from an original 'c' (len 1) or 'cc' (len 2).
            # We add these possibilities to the total count.
            if block_length > 1:
                count += (block_length - 1)

            # Move the index 'i' to the start of the next block.
            i = j

        return count