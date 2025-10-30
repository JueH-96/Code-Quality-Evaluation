class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for c in s:
            # If the top of the stack plus current char form "AB" or "CD", remove them
            if stack and ((stack[-1] == 'A' and c == 'B') or (stack[-1] == 'C' and c == 'D')):
                stack.pop()
            else:
                stack.append(c)
        # Remaining stack size is the minimal possible length
        return len(stack)