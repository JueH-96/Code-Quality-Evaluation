class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        t = 1
        while True:
            m = n - t * k
            if m <= 0:
                return t
            if word.endswith(word[:m]):
                return t
            t += 1