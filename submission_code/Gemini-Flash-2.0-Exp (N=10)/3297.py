class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        if k >= n:
            return 1
        
        for time in range(1, n + 1):
            shifted_word = word
            for _ in range(time):
                shifted_word = shifted_word[k:] + shifted_word[:k]
                
            if shifted_word == word:
                return time