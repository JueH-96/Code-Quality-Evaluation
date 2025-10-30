class Solution:
    def possibleStringCount(self, word: str) -> int:
        total = 0
        current_char = word[0]
        count = 1
        n = len(word)
        
        for i in range(1, n):
            if word[i] == current_char:
                count += 1
            else:
                if count > 1:
                    total += (count - 1)
                current_char = word[i]
                count = 1
        if count > 1:
            total += (count - 1)
        
        return total + 1