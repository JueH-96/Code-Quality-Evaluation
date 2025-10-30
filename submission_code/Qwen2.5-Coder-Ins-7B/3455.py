class Solution:
    def minimumLength(self, s: str) -> int:
        stack = []
        for char in s:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        return len(stack)