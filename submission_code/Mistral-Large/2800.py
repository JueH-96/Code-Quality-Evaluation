class Solution:
    def minLength(self, s: str) -> int:
        stack = []

        for char in s:
            stack.append(char)
            # Check the last two characters in the stack
            if len(stack) >= 2:
                if ''.join(stack[-2:]) == "AB" or ''.join(stack[-2:]) == "CD":
                    stack.pop()
                    stack.pop()

        return len(stack)