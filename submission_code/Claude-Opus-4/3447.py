class Solution:
    def clearDigits(self, s: str) -> str:
        stack = []
        
        for char in s:
            if char.isdigit():
                # Remove the closest non-digit character to the left
                if stack:
                    stack.pop()
            else:
                # Add non-digit character to stack
                stack.append(char)
        
        return ''.join(stack)