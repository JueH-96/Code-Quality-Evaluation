class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for ch in s:
            stack.append(ch)
            # Check if the last two characters form "AB" or "CD"
            if len(stack) >= 2:
                # If top two form "AB" or "CD", pop them out
                if "".join(stack[-2:]) in ("AB", "CD"):
                    stack.pop()
                    stack.pop()
        return len(stack)