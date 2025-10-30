class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        operations = 0
        i = 0
        # Iterate through the string, checking adjacent characters from left to right.
        while i < n - 1:
            # Check if word[i] and word[i+1] are almost-equal.
            # Two characters c1 and c2 are almost-equal if abs(ord(c1) - ord(c2)) <= 1.
            if abs(ord(word[i]) - ord(word[i+1])) <= 1:
                # If they are almost-equal, we must perform at least one operation
                # to resolve this specific conflict.
                # The greedy strategy is to change word[i+1]. This costs 1 operation.
                operations += 1
                # By changing word[i+1], the conflict between word[i] and word[i+1] is resolved.
                # Since we can change word[i+1] to any character, we can pick one
                # that does not conflict with the character at index i (the original word[i])
                # AND also does not conflict with the character at index i+2 (the original word[i+2]).
                # Thus, this single operation effectively resolves the potential conflicts
                # involving the character at index i+1 with both its neighbors.
                # Therefore, after performing an operation on word[i+1], we know that
                # the pair (i, i+1) is resolved, and the pair (i+1, i+2) is also resolved
                # by our choice of the new character at i+1.
                # We can safely skip the next index (i+1) in our iteration and continue checking
                # from the pair starting at index i+2.
                i += 2
            else:
                # If word[i] and word[i+1] are not almost-equal, this pair is fine.
                # No operation is needed for this specific pair.
                # Move to the next adjacent pair, which starts at index i+1.
                i += 1
        return operations