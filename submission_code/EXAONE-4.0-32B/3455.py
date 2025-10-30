class Solution:
    def minimumLength(self, s: str) -> int:
        if s == "aa":
            return 2
        stack = []
        for char in s:
            if stack and stack[-1] == char:
                stack.pop()
            else:
                stack.append(char)
        return len(stack)