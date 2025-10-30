class Solution:
    def possibleStringCount(self, word: str) -> int:
        if not word:
            return 0
        
        total = 1  # Case where no mistakes are made
        prev = word[0]
        count = 1
        
        for char in word[1:]:
            if char == prev:
                count += 1
            else:
                if count > 1:
                    total += (count - 1)
                prev = char
                count = 1
        
        # Add the last group
        if count > 1:
            total += (count - 1)
        
        return total