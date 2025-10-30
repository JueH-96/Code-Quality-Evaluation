class Solution:
    def possibleStringCount(self, word: str) -> int:
        if not word:
            return 0  # though constraints say word is non-empty
        
        runs = []
        prev = word[0]
        current_length = 1
        
        for c in word[1:]:
            if c == prev:
                current_length += 1
            else:
                runs.append(current_length)
                prev = c
                current_length = 1
        runs.append(current_length)  # add the last run
        
        total = 1  # the original word itself
        for l in runs:
            total += (l - 1)
        
        return total