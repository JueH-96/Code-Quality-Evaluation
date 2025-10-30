class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        if k == n:
            return 1  # If k equals the length of the word, one operation will restore it
        
        # We simulate the process of removing and adding characters
        current = word
        for t in range(1, n + 1):  # We need at most n operations (since n is the length of the word)
            # Remove the first k characters and append them to the end
            current = current[k:] + current[:k]
            if current == word:
                return t
        
        return n  # In the worst case, it should return within n steps (though it should always find it sooner)