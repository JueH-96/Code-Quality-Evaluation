class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for char in s:
            if char == 'A' or char == 'B':
                stack.append(char)
            else:
                if stack and stack[-1] == 'A' and char == 'B':
                    stack.pop()
                elif stack and stack[-1] == 'C' and char == 'D':
                    stack.pop()
        return len(stack)