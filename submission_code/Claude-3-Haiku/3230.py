class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        operations = 0
        
        # Create a list to store the characters and their indices
        chars = [(word[i], i) for i in range(n)]
        
        # Sort the list based on the characters
        chars.sort()
        
        # Iterate through the sorted list
        for i in range(1, n):
            # Check if the current character is almost-equal to the previous character
            if abs(ord(chars[i][0]) - ord(chars[i-1][0])) <= 1:
                # Find the index of the character that needs to be changed
                idx = chars[i][1]
                
                # Change the character to a different lowercase letter
                for c in range(ord('a'), ord('z')+1):
                    if chr(c) != word[idx] and abs(c - ord(word[idx])) > 1:
                        word = word[:idx] + chr(c) + word[idx+1:]
                        operations += 1
                        break
        
        return operations