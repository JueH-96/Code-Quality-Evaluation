class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        
        # We only need to check T in [1..n], because the remainder r=(T*k)%n
        # will repeat after n steps at most.
        for T in range(1, n+1):
            r = (T * k) % n
            # We want the substring word[r:] to match word[:n-r].
            # If r == 0, then word[r:] is the entire string, and word[:n-r] is also the entire string.
            if word[r:] == word[:n-r]:
                return T
        
        # In theory, we will always find a solution by T = n, so we should never get here.
        return n