class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for char in s:
            stack.append(char)
            if len(stack) >= 2:
                # Check the last two characters
                if "".join(stack[-2:]) in ("AB", "CD"):
                    # Remove them
                    stack.pop()
                    stack.pop()
        return len(stack)