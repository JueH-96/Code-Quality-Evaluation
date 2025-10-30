class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for char in s:
            if stack:
                top = stack[-1]
                if top + char == "AB" or top + char == "CD":
                    stack.pop()
                    continue
            stack.append(char)
        return len(stack)