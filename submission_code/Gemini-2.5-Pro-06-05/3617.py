class Solution:
    def possibleStringCount(self, word: str) -> int:
        """
        Calculates the total number of possible original strings Alice might have intended to type.

        The total number of possibilities is the sum of:
        1. The case where no mistake was made (the original string is `word` itself). This is 1 possibility.
        2. The cases where exactly one long-press mistake was made. A run of identical characters
           of length L > 1 in `word` could have been formed from L-1 shorter original runs.
        """

        # Start with 1 to account for the case where `word` itself is the intended string (no long press).
        count = 1
        
        n = len(word)
        i = 0
        while i < n:
            # Find the end of the current run of identical characters.
            j = i
            while j + 1 < n and word[j + 1] == word[i]:
                j += 1
            
            # The length of the current run is (j - i + 1).
            run_length = j - i + 1
            
            # If the run's length is greater than 1, it could have been formed
            # by a long press on a shorter original string. This adds `run_length - 1`
            # new possibilities.
            if run_length > 1:
                count += run_length - 1
            
            # Move the main pointer to the beginning of the next run.
            i = j + 1
            
        return count