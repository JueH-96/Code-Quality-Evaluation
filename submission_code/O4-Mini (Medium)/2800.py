class Solution:
    def minLength(self, s: str) -> int:
        # Use a stack to greedily remove "AB" or "CD" as they appear
        stack = []
        for ch in s:
            if stack:
                top = stack[-1]
                # If the top of the stack and current char form "AB" or "CD", pop the top
                if (top == 'A' and ch == 'B') or (top == 'C' and ch == 'D'):
                    stack.pop()
                    continue
            # Otherwise, push the current char onto the stack
            stack.append(ch)
        # The remaining stack size is the minimal possible length
        return len(stack)