class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        length = len(word)
        word = word + word
        for i in range(1, length+1):
            if word[i:i+length] == word[:length]:
                return i
        return length