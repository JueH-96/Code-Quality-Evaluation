class Solution:
    def maxOperations(self, s: str) -> int:
        # Initialize the count of operations
        operations = 0
        # Initialize the stack
        stack = []
        # Iterate over the string
        for c in s:
            # If the stack is not empty and the top element is '1' and the current character is '0'
            if stack and stack[-1] == '1' and c == '0':
                # Pop the top element from the stack
                stack.pop()
                # Increment the count of operations
                operations += 1
            else:
                # Otherwise, push the current character to the stack
                stack.append(c)
        # Return the count of operations
        return operations