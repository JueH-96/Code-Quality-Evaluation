class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        
        # Try each possible number of steps
        for t in range(1, (n + k - 1) // k + 1):
            # After t steps, we've removed t*k characters
            removed = t * k
            
            if removed >= n:
                # If we've removed all characters, we can always restore
                return t
            
            # Check if the remaining suffix matches the prefix of original word
            remaining_suffix = word[removed:]
            if word.startswith(remaining_suffix):
                return t
        
        # If no solution found in the loop, return the maximum possible steps
        return (n + k - 1) // k