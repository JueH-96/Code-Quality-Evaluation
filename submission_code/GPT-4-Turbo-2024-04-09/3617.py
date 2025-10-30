class Solution:
    def possibleStringCount(self, word: str) -> int:
        if not word:
            return 0
        
        count = 1  # Always include the original string
        n = len(word)
        
        for i in range(1, n):
            if word[i] == word[i - 1]:
                # If current character is the same as the previous one,
                # it's a potential spot where a character might have been overtyped.
                count += 1
            else:
                # Reset the consecutive character count on change of character
                if count > 1:
                    break
        
        # Check if the entire string is made of the same character
        if count > 1 and all(word[i] == word[0] for i in range(n)):
            return count
        
        return count if count > 1 else 1