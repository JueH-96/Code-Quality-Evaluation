class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        # If the word has 1 or 0 characters, no adjacent pairs exist.
        if n <= 1:
            return 0

        operations = 0
        i = 1
        while i < n:
            # Two characters are almost-equal if their ASCII values differ by at most 1.
            if abs(ord(word[i]) - ord(word[i - 1])) <= 1:
                # We found an adjacent pair that is almost-equal.
                # We must perform one operation. The greedy choice is to change
                # the character at index `i`. This resolves the conflict with `word[i-1]`.
                # Since we can change `word[i]` to any character, we can also ensure
                # it won't conflict with `word[i+1]`.
                operations += 1
                
                # Because we've conceptually changed `word[i]`, the pair (i, i+1)
                # is resolved. We can skip the character at `i` and the next check
                # will be at `i+2`.
                i += 2
            else:
                # The pair (i-1, i) is fine. Move to the next character to check.
                i += 1
        
        return operations