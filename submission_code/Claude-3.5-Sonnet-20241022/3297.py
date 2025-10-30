class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        seconds = 1
        
        while seconds * k < n:
            if word[seconds * k:] == word[:n - seconds * k]:
                return seconds
            seconds += 1
            
        return seconds