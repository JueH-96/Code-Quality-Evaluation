class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for char in s:
            stack.append(char)
            while len(stack) >= 2:
                top = stack[-1]
                second_top = stack[-2]
                if (second_top == 'A' and top == 'B') or (second_top == 'C' and top == 'D'):
                    stack.pop()
                    stack.pop()
                else:
                    break
        return len(stack)