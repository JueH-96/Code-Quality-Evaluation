class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        time = 1
        while True:
            if word.startswith(word[time * k:]):
                return time
            time += 1