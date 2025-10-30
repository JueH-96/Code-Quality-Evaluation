class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        operations = 0
        n = len(word)
        
        for i in range(1, n):
            if abs(ord(word[i]) - ord(word[i - 1])) <= 1:
                operations += 1
                # Change the current character to a non-adjacent character
                if word[i] == 'a':
                    word = word[:i] + 'c' + word[i + 1:]
                elif word[i] == 'z':
                    word = word[:i] + 'x' + word[i + 1:]
                else:
                    word = word[:i] + chr(ord(word[i]) + 2) + word[i + 1:]
        
        return operations