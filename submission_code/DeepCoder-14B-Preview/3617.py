class Solution:
    def possibleStringCount(self, word: str) -> int:
        if not word:
            return 0
        
        # Split the word into runs of consecutive characters
        runs = []
        current_char = word[0]
        current_run_length = 1
        
        for c in word[1:]:
            if c == current_char:
                current_run_length += 1
            else:
                runs.append(current_run_length)
                current_char = c
                current_run_length = 1
        runs.append(current_run_length)  # Add the last run
        
        # Calculate the total number of possible original strings
        total = 1  # The case where no characters are removed
        for run_length in runs:
            if run_length >= 2:
                total += 1
        
        return total