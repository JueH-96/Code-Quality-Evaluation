class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        
        for char in s:
            if stack and ((stack[-1] == 'A' and char == 'B') or (stack[-1] == 'C' and char == 'D')):
                stack.pop()  # Remove previous character if it forms "AB" or "CD" with current
            else:
                stack.append(char)
                
        return len(stack)