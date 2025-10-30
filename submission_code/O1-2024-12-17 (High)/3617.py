class Solution:
    def possibleStringCount(self, word: str) -> int:
        # Count consecutive runs of the same character
        runs = []
        current_char = word[0]
        current_count = 1
        
        for i in range(1, len(word)):
            if word[i] == current_char:
                current_count += 1
            else:
                runs.append(current_count)
                current_char = word[i]
                current_count = 1
        runs.append(current_count)  # append the last run
        
        # The formula: 1 (no inflation) + sum of (run_length - 1) for each run
        return 1 + sum(run_length - 1 for run_length in runs)