class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        original_word = word
        current_word = word
        time = 0
        for t in range(1, len(word) + 1):
            current_word = current_word[k:] + current_word[:k]
            if current_word == original_word:
                return t
        return -1