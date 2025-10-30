import collections # This import is not strictly necessary as we use a list, but useful to know deque exists

class Solution:
    def minLength(self, s: str) -> int:
        """
        Calculates the minimum possible length of the string after repeatedly
        removing "AB" or "CD" substrings using a stack-based approach.

        The core idea is to process the string character by character, maintaining
        a stack that represents the current state of the processed string after
        removals. When a new character arrives, if it forms a removable pair 
        ("AB" or "CD") with the character at the top of the stack, we remove 
        the pair by popping the top character from the stack and not adding the 
        current character. Otherwise, we push the current character onto the stack.
        The final length of the stack gives the minimum possible length of the string.

        Args:
            s: The input string consisting of uppercase English letters.
               Constraints: 1 <= len(s) <= 100. s consists only of uppercase English letters.

        Returns:
            The minimum possible length of the resulting string.
        """
        stack = [] # Use a standard Python list as a stack

        # Process each character of the string sequentially
        for char in s:
            # Check if the stack is non-empty. If it is, check if the top element
            # combined with the current character forms a removable pair ("AB" or "CD").
            # The `stack[-1]` accesses the top element of the stack (last element of the list).
            if stack and (stack[-1] + char == "AB" or stack[-1] + char == "CD"):
                # If a removable pair ("AB" or "CD") is formed by the top of the stack
                # and the current character, pop the top element from the stack.
                # This simulates the removal of the substring. We don't push the 
                # current character 'char' because it completes the pair being removed.
                stack.pop()
            else:
                # If the stack is empty, or if the top element and the current character
                # do not form a removable pair, push the current character onto the stack.
                stack.append(char)

        # After iterating through all characters in the input string 's', 
        # the 'stack' contains the characters of the resulting string after
        # all possible "AB" and "CD" removals have been performed greedily.
        # The length of the stack is the minimum possible length of the string.
        return len(stack)