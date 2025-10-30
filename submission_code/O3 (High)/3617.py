class Solution:
    def possibleStringCount(self, word: str) -> int:
        """
        The final string `word` is obtained from the intended string `S`
        by possibly choosing **one** of the typed characters and repeating
        it some positive number of extra times (long-press).  All the other
        characters appear exactly once.

        For every maximal run of equal characters in `word`
             run length = L
        • if this run is NOT the long-pressed one, its length in `S`
          must also be L.
        • if this run IS the long-pressed one, its length in `S`
          can be any value from 1 … L-1 (at least one duplicate was added).

        So for a run of length L there are (L-1) possibilities to be the
        long-pressed run, and for the whole word we have

              1  (no long-press at all)
            + Σ (L_i – 1)   over all runs i

        Note that Σ L_i = len(word) and the number of runs is `runs`,
        therefore the formula simplifies to

              1 + len(word) – runs
        """
        if not word:                       # (problem guarantees non-empty)
            return 0

        runs = 1                           # at least one run exists
        for i in range(1, len(word)):
            if word[i] != word[i - 1]:
                runs += 1                  # new run starts

        return 1 + len(word) - runs