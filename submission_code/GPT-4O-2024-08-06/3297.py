class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        # The length of the word
        n = len(word)
        
        # We need to find the smallest t > 0 such that after t operations, the word is back to its initial state.
        # Each operation is equivalent to a cyclic shift of the word by k positions.
        
        # We will simulate the process and check when the word returns to its initial state.
        current_word = word
        for t in range(1, n + 1):
            # Perform the operation: remove first k characters and append them to the end
            current_word = current_word[k:] + current_word[:k]
            
            # Check if the word is back to its initial state
            if current_word == word:
                return t
        
        # If we never return to the initial state within n operations, something is wrong
        # but according to the problem constraints, we should always find a solution.
        return -1