class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        for time in range(1, n + 1):
            # Remove the first k characters and add the same characters to the end
            new_word = word[time * k:] + word[:time * k]
            if new_word == word:
                return time
        return -1  # This line should not be reached according to the problem constraints