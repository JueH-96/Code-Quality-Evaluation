class Solution:
    def clearDigits(self, s: str) -> str:
        stack = []
        for char in s:
            if char.isdigit():
                # Remove the closest non-digit character (top of the stack)
                if stack:  
                    stack.pop()
                # Skip adding the digit to the stack, effectively removing it
            else:
                # Push non-digit characters onto the stack
                stack.append(char)
        return "".join(stack)