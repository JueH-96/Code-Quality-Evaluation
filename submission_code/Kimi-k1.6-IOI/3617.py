class Solution:
    def possibleStringCount(self, word: str) -> int:
        if not word:
            return 0
        
        groups = []
        current_char = word[0]
        count = 1
        
        for c in word[1:]:
            if c == current_char:
                count += 1
            else:
                groups.append((current_char, count))
                current_char = c
                count = 1
        groups.append((current_char, count))
        
        total = 1  # account for the original string itself
        for _, cnt in groups:
            total += (cnt - 1)
        
        return total