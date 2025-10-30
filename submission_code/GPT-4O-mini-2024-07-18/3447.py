class Solution:
    def clearDigits(self, s: str) -> str:
        stack = []
        
        for char in s:
            if char.isdigit():
                if stack:  # Only pop if there's something in the stack
                    stack.pop()  # Remove the last non-digit character
            else:
                stack.append(char)  # Add non-digit characters to the stack
        
        return ''.join(stack)