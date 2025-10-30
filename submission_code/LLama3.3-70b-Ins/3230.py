class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        stack = []
        count = 0
        
        for char in word:
            if stack and (char == stack[-1] or abs(ord(char) - ord(stack[-1])) == 1):
                count += 1
            else:
                stack.append(char)
        
        return count