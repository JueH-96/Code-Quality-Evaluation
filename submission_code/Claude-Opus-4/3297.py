class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        
        for t in range(1, n):
            # Check if we've removed all characters
            if t * k >= n:
                return t
            
            # Check if the remaining suffix matches the beginning of word
            if word[t * k:] == word[:n - t * k]:
                return t
        
        # If no match found, return ceil(n/k)
        return (n + k - 1) // k