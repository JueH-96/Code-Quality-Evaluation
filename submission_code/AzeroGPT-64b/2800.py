class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for i in s:
            if stack and stack[-1] + i in ('AB', 'CD'):
                stack.pop()
            else:
                stack.append(i)
        return len(stack)