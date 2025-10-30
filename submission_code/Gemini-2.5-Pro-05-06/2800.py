class Solution:
    def minLength(self, s: str) -> int:
        stack = []  # Use a Python list as a stack
        
        for char_val in s:
            stack.append(char_val)
            
            # After appending, check if a reduction is possible
            # The stack must have at least 2 characters to form "AB" or "CD"
            if len(stack) >= 2:
                # Check for "AB"
                if stack[-2] == 'A' and stack[-1] == 'B':
                    stack.pop()  # Remove 'B'
                    stack.pop()  # Remove 'A'
                # Else, check for "CD" (only if "AB" was not formed/removed)
                elif stack[-2] == 'C' and stack[-1] == 'D':
                    stack.pop()  # Remove 'D'
                    stack.pop()  # Remove 'C'
                    
        return len(stack)