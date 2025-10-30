class Solution:
    def minLength(self, s: str) -> int:
        """
        Calculates the minimum possible length of a string by repeatedly removing
        "AB" or "CD" substrings using a stack-based approach.
        """
        stack = []
        
        for char in s:
            # Check if the stack is not empty and if the top element
            # can form a removable pair ("AB" or "CD") with the current character.
            if stack and ((stack[-1] == 'A' and char == 'B') or 
                          (stack[-1] == 'C' and char == 'D')):
                # If a pair is found, it means a substring can be removed.
                # We simulate this by popping the previous character from the stack
                # and not adding the current one.
                stack.pop()
            else:
                # Otherwise, the current character is added to the stack,
                # as it doesn't form a removable pair with the end of the
                # processed string.
                stack.append(char)
                
        # The final size of the stack represents the length of the
        # irreducible string.
        return len(stack)