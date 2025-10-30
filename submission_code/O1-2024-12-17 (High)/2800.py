class Solution:
    def minLength(self, s: str) -> int:
        stack = []
        for c in s:
            # Check if the top of the stack and the new character form "AB" or "CD"
            if stack and stack[-1] + c in ("AB", "CD"):
                stack.pop()
            else:
                stack.append(c)
        # The remaining stack size is the minimum possible length
        return len(stack)