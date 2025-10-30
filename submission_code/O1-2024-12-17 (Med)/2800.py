class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for char in s:
            stack.append(char)
            # Whenever the top two characters are "AB" or "CD", remove them
            if len(stack) >= 2 and (stack[-2] + stack[-1] in ["AB", "CD"]):
                stack.pop()
                stack.pop()
        return len(stack)