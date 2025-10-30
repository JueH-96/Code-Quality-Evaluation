class Solution:
    def possibleStringCount(self, word: str) -> int:
        # We can have at most one long‐press mistake on a single character run.
        # If a run has length L > 1, reducing that run to any length 1..L
        # (except the case 1..L==L which is no change) yields (L−1) distinct originals.
        # Summing over all runs and adding 1 for the "no mistake at all" case gives the result.
        
        if not word:
            return 0
        
        ans = 1  # the case of no long‐press mistake
        i = 0
        n = len(word)
        while i < n:
            j = i+1
            # find end of current run
            while j < n and word[j] == word[i]:
                j += 1
            run_length = j - i
            if run_length > 1:
                ans += (run_length - 1)
            i = j
        
        return ans