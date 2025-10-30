class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        # We need to find the smallest t > 0 such that the word returns to its original state after t operations.
        # Each operation removes the first k characters and adds any k characters to the end.
        # The word will return to its original state if the remaining part of the word after removing the first t*k characters is the same as the initial word.
        # So, we need to find the smallest t such that word[t*k:] == word[:n - t*k].
        # Since t*k must be a multiple of k, and t*k <= n, we can iterate t from 1 to n//k.
        for t in range(1, n // k + 1):
            if word[t*k:] == word[:n - t*k]:
                return t
        # If no such t is found, then the word will return to its original state after n//k operations.
        return n // k