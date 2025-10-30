class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for c in s:
            stack.append(c)
            while len(stack) >= 2:
                a, b = stack[-2], stack[-1]
                if (a == 'A' and b == 'B') or (a == 'C' and b == 'D'):
                    stack.pop()
                    stack.pop()
                else:
                    break
        return len(stack)