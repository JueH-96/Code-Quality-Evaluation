class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        for time in range(1, n + 1):
            new_word = word
            for _ in range(time):
                new_word = new_word[k:] + new_word[:k]
            if new_word == word:
                return time
        return -1