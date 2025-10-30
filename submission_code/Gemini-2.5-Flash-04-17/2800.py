class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for char in s:
            # Check if the stack is not empty and the last two characters form "AB" or "CD"
            if stack and ((stack[-1] == 'A' and char == 'B') or (stack[-1] == 'C' and char == 'D')):
                stack.pop() # Remove the pair by popping the last element from the stack
            else:
                stack.append(char) # Otherwise, push the current character onto the stack
                
        # The remaining elements in the stack form the shortest possible string
        return len(stack)