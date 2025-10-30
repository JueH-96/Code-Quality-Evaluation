class Solution:
    def possibleStringCount(self, word: str) -> int:
        # Count runs of consecutive identical characters
        runs = []
        current_char = None
        current_count = 0
        
        for ch in word:
            if ch == current_char:
                current_count += 1
            else:
                if current_char is not None:
                    runs.append(current_count)
                current_char = ch
                current_count = 1
        runs.append(current_count)
        
        # Sum of (length_of_run - 1) for each run where length_of_run > 1
        # plus one for the scenario where no expansion happened
        expansions = sum((r - 1) for r in runs if r > 1)
        return 1 + expansions