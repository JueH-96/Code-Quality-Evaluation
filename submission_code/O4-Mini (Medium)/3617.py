class Solution:
    def possibleStringCount(self, word: str) -> int:
        """
        Count the number of possible original strings given that Alice
        may have long‑pressed at most one character run.
        For each run of length L, if it's the long‑pressed one, the original
        run could have been 1..(L-1) in length. Other runs must match exactly.
        Total ways = 1 (no long‑press) + sum over runs of (L-1).
        Simplifies to len(word) - num_runs + 1.
        """
        if not word:
            return 0
        
        n = len(word)
        # Count runs
        runs = 1
        for i in range(1, n):
            if word[i] != word[i-1]:
                runs += 1
        
        # Total possibilities: 1 (no long-press) + sum(Li - 1) = sum(Li) - runs + 1
        # sum(Li) = n
        return n - runs + 1