class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for c in s:
            if stack:
                last = stack[-1]
                if (last == 'A' and c == 'B') or (last == 'C' and c == 'D'):
                    stack.pop()
                    continue
            stack.append(c)
        return len(stack)