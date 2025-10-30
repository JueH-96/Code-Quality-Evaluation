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
                groups.append(count)
                current_char = c
                count = 1
        groups.append(count)  # Add the last group
        
        total = sum(g - 1 for g in groups) + 1
        return total