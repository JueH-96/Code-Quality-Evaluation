class Solution:
    def clearDigits(self, s: str) -> str:
        stack = []
        for c in s:
            if c.isdigit():
                if stack and not stack[-1].isdigit():
                    stack.pop()
            else:
                stack.append(c)
        return ''.join(stack)