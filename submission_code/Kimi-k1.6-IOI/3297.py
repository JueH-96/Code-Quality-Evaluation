class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        t = 1
        while True:
            if t * k >= n:
                return t
            suffix = word[t * k:]
            if word.startswith(suffix):
                return t
            t += 1