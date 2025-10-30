class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        def almost_equal(c1, c2):
            return c1 == c2 or abs(ord(c1) - ord(c2)) == 1
        
        n = len(word)
        if n <= 1:
            return 0
        
        changes = 0
        i = 0
        while i < n - 1:
            if almost_equal(word[i], word[i + 1]):
                # Find a character that is not almost equal to word[i] or word[i+1]
                # We have 26 letters, and at most 4 are invalid (word[i-1], word[i], word[i+1], word[i+2])
                # So there's always at least one valid choice.
                valid_choices = set(chr(j) for j in range(ord('a'), ord('z') + 1))
                if i > 0:
                    valid_choices.discard(word[i - 1])
                valid_choices.discard(word[i])
                valid_choices.discard(word[i + 1])
                if i + 2 < n:
                    valid_choices.discard(word[i + 2])
                
                # Change word[i+1] to any valid choice (pick the first available)
                word = word[:i + 1] + next(iter(valid_choices)) + word[i + 2:]
                changes += 1
            
            i += 1
        
        return changes