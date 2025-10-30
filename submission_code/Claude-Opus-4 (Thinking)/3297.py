class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        # Maximum time needed is when we remove all characters
        max_t = (n + k - 1) // k  # This is ceil(n/k)
        
        # Check each time from 1 to max_t-1
        for t in range(1, max_t):
            # Check if the remaining suffix matches the required prefix
            if word[t * k:] == word[:n - t * k]:
                return t
        
        # If no earlier match found, return max_t
        return max_t