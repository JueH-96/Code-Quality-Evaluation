class Solution:
    def minimumTime(self, word: str, k: int) -> int:
        n = len(word)
        if k == 0 or k == n:
            return 0  # No change possible
        
        original = word
        current = original
        t = 0
        
        while True:
            # Try all possible added strings of length k
            for added in itertools.product('a'*k, repeat=k):
                added_str = ''.join(added)
                new_word = current[k:] + added_str
                if new_word == original:
                    return t + 1
                # Check if new_word has any chance to cycle back
                if len(new_word) != n:
                    # This should not happen as k is valid
                    continue
            # If no added string brings back the word, try more steps
            t += 1
            current = current[k:] + current[:k]
            if t > n * (n // k + 1):
                # Prevent infinite loop, though problem says it's always possible
                return -1

        return -1