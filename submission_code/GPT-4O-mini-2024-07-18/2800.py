class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        
        for char in s:
            stack.append(char)
            # Check the last two characters in the stack
            if len(stack) >= 2 and ((stack[-2] == 'A' and stack[-1] == 'B') or (stack[-2] == 'C' and stack[-1] == 'D')):
                stack.pop()  # Remove the last character
                stack.pop()  # Remove the second last character
        
        return len(stack)