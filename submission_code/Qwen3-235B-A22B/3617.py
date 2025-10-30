class Solution:
    def possibleStringCount(self, word: str) -> int:
        if not word:
            return 0
        
        runs = []
        prev = word[0]
        count = 1
        
        for c in word[1:]:
            if c == prev:
                count += 1
            else:
                runs.append(count)
                prev = c
                count = 1
        runs.append(count)
        
        total = 0
        for r in runs:
            total += (r - 1)
        
        return total + 1