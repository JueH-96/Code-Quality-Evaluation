class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for c in s:
            if stack and stack[-1] + c == 'AB':
                stack.pop()
            elif stack and stack[-1] + c == 'CD':
                stack.pop()
            else:
                stack.append(c)
        return len(stack)